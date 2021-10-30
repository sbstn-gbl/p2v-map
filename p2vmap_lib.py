import collections
import glob
import itertools
import os
import re
import warnings
from typing import Generic, Tuple, TypeVar

import numpy as np
import pandas as pd
import plotly.graph_objs as go
import sklearn.manifold
import sklearn.model_selection
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.tensorboard
import yaml
from loguru import logger
from plotly.offline import iplot

T = TypeVar("T")

warnings.simplefilter(action="error", category=FutureWarning)


def read_yaml(x: str) -> dict:
    """
    Read yaml files
        x: name of the file
    """
    with open(x, "r") as con:
        config = yaml.safe_load(con)
    return config


class NegativeSamplesGenerator:
    """
    Class for generating negative samples
        get_negative_samples: produce negative samples
    """

    def __init__(
        self,
        data: pd.DataFrame,
        n_negative_samples: int,
        batch_size: int,
        power: float = 0.75,
        variable_product: str = "product",
    ):
        """
        Initialize negative samples generator (for given positive samples)
            data: basket data, must contain `variable_basket` and `variable_product`
            n_negative_samples: number of negative samples per positive sample
            batch_size: size of a single batch
            power: distortion factor for negative sample generator
            variable_product: product identifier in `data`
        """

        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size
        self.power = power
        self.n_draws = self.batch_size * self.n_negative_samples
        self.variable_product = variable_product
        self.domain = (2 ** 31 - 1,)
        self._build_product_counts(data)
        self._build_cumulative_count_table()
        self.products = np.array(list(self.counts.keys()))

    def get_negative_samples(self, context: np.ndarray = None) -> np.ndarray:
        """
        Produce negative samples (for given positive samples)
            context: context products that may not be used as negative samples
        """
        if context is not None:
            negative_samples = (
                np.zeros((self.n_negative_samples, len(context)), dtype=np.int32) - 1
            )
            done_sampling = False
            while not done_sampling:
                new_sample_index = negative_samples == -1
                n_draws = np.sum(new_sample_index)
                random_integers = np.random.randint(0, self.domain, n_draws)
                new_negative_samples_index = np.searchsorted(
                    self.cumulative_count_table, random_integers
                )
                new_negative_samples = self.products[new_negative_samples_index]
                negative_samples[new_sample_index] = new_negative_samples
                negative_samples[negative_samples == context] = -1
                done_sampling = np.all(negative_samples != -1)
            return negative_samples
        else:
            random_integers = np.random.randint(0, self.domain, self.n_draws)
            negative_samples_index = np.searchsorted(
                self.cumulative_count_table, random_integers
            )
            return self.products[negative_samples_index].reshape(
                (self.batch_size, self.n_negative_samples)
            )

    def _build_product_counts(self, x: pd.DataFrame) -> None:
        """
        Count number of times products occur in basket data
        """
        n_products = x[self.variable_product].max() + 1
        product_counts = (
            x.groupby(self.variable_product)[self.variable_product].count().to_dict()
        )
        product_counts_filled = collections.OrderedDict()
        for j in range(n_products):
            if j not in product_counts:
                product_counts_filled[j] = 0
            else:
                product_counts_filled[j] = product_counts[j]
        self.counts = product_counts_filled

    def _build_cumulative_count_table(self) -> None:
        """
        Build count table (mapped to self.domain) for integer sampling of products
        """
        tmp = np.array(list(self.counts.values())) ** self.power
        cumulative_relative_count_table = np.cumsum(tmp / sum(tmp))
        self.cumulative_count_table = np.int32(
            (cumulative_relative_count_table * self.domain).round()
        )
        assert self.cumulative_count_table[-1] == self.domain


class DataStreamP2V:
    """
    Class for generating P2V training samples
        generate_batch: produce a batch of training samples
        reset_iterator: reset data streamer and empty sample cache
    """

    def __init__(
        self,
        data: pd.DataFrame,
        variable_basket: str,
        variable_product: str,
        batch_size: int = 8_192,
        shuffle: bool = True,
        n_negative_samples: int = 0,
        power: float = 0.75,
        allow_context_collisions: bool = False,
    ):
        """
        Initialize P2V data streamer
            data: must contain `variable_basket` and `variable_product`
            variable_basket: basket identifier in `data`
            variable_product: product identifier in `data`
            batch_size: size of a single batch
            shuffle: shuffle data when resetting streamer
            n_negative_samples: number of negative samples per positive sample
            power: distortion factor for negative sample generator
            allow_context_collisions: allow that an id is a positive and a negative sample at the same time
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.cached_samples = []
        self.basket_list = self._basket_df_to_list(
            x=data, variable_basket=variable_basket, variable_product=variable_product
        )
        self.reset_iterator()
        self.produce_negative_samples = n_negative_samples > 0
        if self.produce_negative_samples:
            self.allow_context_collisions = allow_context_collisions
            self.negative_samples_generator = NegativeSamplesGenerator(
                data=data,
                n_negative_samples=n_negative_samples,
                batch_size=self.batch_size,
                power=power,
                variable_product=variable_product,
            )

    def generate_batch(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Produce a batch of training samples containing center products, context products, and negative samples
        """
        # fill cache
        self._fill_cache()

        # generate skip-gram pairs
        output_array = np.asarray(self.cached_samples[: self.batch_size])
        self.cached_samples = self.cached_samples[self.batch_size :]
        center = output_array[:, 0, 0].astype(np.int64)
        context = output_array[:, 1, 0].astype(np.int64)

        # add negative samples
        if self.produce_negative_samples:
            if self.allow_context_collisions:
                negative_samples = (
                    self.negative_samples_generator.get_negative_samples()
                )
            else:
                negative_samples = self.negative_samples_generator.get_negative_samples(
                    context
                ).T
        else:
            negative_samples = np.empty(shape=(2, 0))

        # return
        return center, context, negative_samples

    def reset_iterator(self) -> None:
        """
        Reset data streamer and empty sample cache
        """
        if self.shuffle:
            np.random.shuffle(self.basket_list)
        self.basket_iterator = self._basket_iterator(self.basket_list)
        self.cached_samples = []

    def _basket_df_to_list(self, x, variable_basket, variable_product):
        """
        Turn a basket dataframe into a list of baskets
        """
        x_basket_values = (
            x[[variable_basket, variable_product]].sort_values([variable_basket]).values
        )
        keys = x_basket_values[:, 0]
        ukeys, index = np.unique(keys, True)
        return np.split(x_basket_values[:, 1:], index)[1:]

    def _basket_iterator(self, basket_list):
        """
        Iterator yielding single baskets
        """
        for basket in basket_list:
            yield basket

    def _fill_cache(self):
        """
        Fill sample cache with center-context pairs
        """
        fill_cache = len(self.cached_samples) < self.batch_size
        while fill_cache:
            try:
                new_basket = next(self.basket_iterator, None)
                self.cached_samples.extend(itertools.permutations(new_basket, 2))
            except:
                fill_cache = False
            if len(self.cached_samples) >= self.batch_size:
                fill_cache = False


def build_data_loader(
    streamer,
    config_train,
    config_validation,
    validation_size,
):
    # build numpy arrays for full dataset
    streamer.reset_iterator()
    list_ce = []
    list_co = []
    list_ns = []
    while True:
        try:
            ce, co, ns = streamer.generate_batch()
            list_ce.append(ce)
            list_co.append(co)
            list_ns.append(ns)
        except:
            break
    ce = np.hstack(list_ce)
    co = np.hstack(list_co)
    ns = np.vstack(list_ns)

    # train/validation split
    ce_t, ce_v, co_t, co_v, ns_t, ns_v = sklearn.model_selection.train_test_split(
        ce,
        co,
        ns,
        test_size=validation_size,
    )

    # build data loader
    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.LongTensor(ce_t),
            torch.LongTensor(co_t),
            torch.LongTensor(ns_t),
        ),
        **config_train,
    )
    dl_val = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.LongTensor(ce_v),
            torch.LongTensor(co_v),
            torch.LongTensor(ns_v),
        ),
        **config_validation,
    )
    return dl_train, dl_val


class P2V(torch.nn.Module):
    def __init__(self, n_products, size):
        super().__init__()

        # trainable variables
        self.wi = torch.nn.Embedding(n_products, size, sparse=True)
        with torch.no_grad():
            self.wi.weight.uniform_(-0.025, 0.025)
        self.wo = torch.nn.Embedding(n_products, size, sparse=True)
        with torch.no_grad():
            self.wo.weight.uniform_(-0.025, 0.025)

    def forward(self, center, context, negative_samples):

        # embed products (center, context, negative_samples)
        wi_center = self.wi(center)
        wo_positive_samples = self.wo(context)
        wo_negative_samples = self.wo(negative_samples)

        # logits
        logits_positive_samples = torch.einsum(
            "ij,ij->i", (wi_center, wo_positive_samples)
        )
        logits_negative_samples = torch.einsum(
            "ik,ijk->ij", (wi_center, wo_negative_samples)
        )

        # loss
        loss_positive_samples = F.binary_cross_entropy_with_logits(
            input=logits_positive_samples,
            target=torch.ones_like(logits_positive_samples),
            reduction="sum",
        )
        loss_negative_samples = F.binary_cross_entropy_with_logits(
            input=logits_negative_samples,
            target=torch.zeros_like(logits_negative_samples),
            reduction="sum",
        )

        n_samples = logits_positive_samples.shape[0] * (
            logits_negative_samples.shape[1] + 1
        )
        return (loss_positive_samples + loss_negative_samples) / n_samples


class TrainerP2V:
    def __init__(self, model, train, validation, path, n_batch_log=500):
        self.model = model
        self.train = train
        self.validation = validation
        self.optimizer = torch.optim.SparseAdam(params=list(model.parameters()))
        self.path = path
        os.makedirs(f"{self.path}/weights", exist_ok=True)
        self.writer_train = torch.utils.tensorboard.SummaryWriter(f"{self.path}/train")
        self.writer_val = torch.utils.tensorboard.SummaryWriter(f"{self.path}/val")
        self.n_batch_log = n_batch_log
        self.global_batch = 0
        self.epoch = 0
        self.batch = 0

    def fit(self, epochs):

        for _ in range(epochs):
            print(f"epoch = {self.epoch}")

            for ce, co, ns in self.train:
                self.batch += 1
                self.global_batch += 1
                self.optimizer.zero_grad()
                loss_train = self.model(ce, co, ns)
                loss_train.backward()
                self.optimizer.step()
                self.writer_train.add_scalar("loss", loss_train, self.global_batch)

                if self.batch % self.n_batch_log == 1:
                    self._callback_batch()

            self.epoch += 1

        self.writer_train.flush()
        self.writer_train.close()
        self.writer_val.flush()
        self.writer_val.close()

    def _callback_batch(self):
        # validation loss
        self.model.eval()
        with torch.no_grad():
            list_loss_validation = []
            for ce, co, ns in self.validation:
                list_loss_validation.append(self.model(ce, co, ns).item())
            loss_validation = np.mean(list_loss_validation)
        self.writer_val.add_scalar("loss", loss_validation, self.global_batch)
        self.model.train()

        # save input embedding
        np.save(
            f"{self.path}/weights/wi_{self.epoch:02d}_{self.batch:06d}.npy",
            self.get_wi(),
        )

    def predict(self):
        None

    def get_wi(self):
        return self.model.wi.weight.detach().numpy()

    def get_wo(self):
        return self.model.wo.weight.detach().numpy()


class DashboardP2V(object):
    def __init__(self, path, n_steps=20):
        self.path = path
        self.n_steps = n_steps
        self.files = {
            "wi": "wi_*.npy",
        }
        self.file_df = None
        self.plot_data = {}
        self.plot_data["steps"] = []
        self.plot_data["wi"] = []
        self.plot_data["tsne"] = []
        self.raw_files = glob.glob(f"{self.path}/{self.files['wi']}")

    def plot_product_embedding(
        self,
        idx=None,
        label="wi",
        size=None,
        l2norm=True,
        transpose=True,
        reload=True,
    ):
        if reload:
            self._load_data(idx=idx, agg=None)
        data = []
        steps = []
        for i in range(self.n_steps):
            data_i = self.plot_data[label][i]
            # data
            if size is not None:
                data_i = data_i.reshape(size)
            if l2norm:
                data_i /= np.linalg.norm(data_i, axis=1)[:, np.newaxis]
            if transpose:
                data_i = data_i.T
            data.append(go.Heatmap(z=data_i, colorscale="Jet", zmin=-0.6, zmax=0.6))
            # step
            step = dict(
                method="restyle",
                label=self.plot_data["steps"][i],
                args=["visible", [False] * self.n_steps],
            )
            step["args"][1][i] = True
            steps.append(step)
        sliders = dict(
            active=0, currentvalue={"visible": False}, pad={"t": 50}, steps=steps
        )
        layout = go.Layout(
            height=700,
            width=1000,
            sliders=[sliders],
            margin=go.layout.Margin(l=50, r=50, b=150, t=20, pad=4),
            template="plotly_white",
        )
        return iplot(dict(data=data, layout=layout))

    def plot_tsne_map(self, product, config):
        if len(self.plot_data["tsne"]) < self.n_steps:
            self._tsne(product, config)
        data = []
        steps = []
        for i in range(self.n_steps):
            data_tsne_i = self.plot_data["tsne"][i]
            trace = go.Scatter(
                x=data_tsne_i["x"].values,
                y=data_tsne_i["y"].values,
                text=[
                    f"c: {c} <br> j: {c}"
                    for (c, j) in zip(
                        data_tsne_i["product"].values,
                        data_tsne_i["category"].values,
                    )
                ],
                hoverinfo="text",
                mode="markers",
                marker=dict(
                    size=12,
                    color=data_tsne_i["category"].values,
                    colorscale="Jet",
                    showscale=False,
                ),
            )
            data.append(trace)
            # step
            step = dict(
                method="restyle",
                label=self.plot_data["steps"][i],
                args=["visible", [False] * self.n_steps],
            )
            step["args"][1][i] = True
            steps.append(step)
        sliders = dict(
            active=0, currentvalue={"visible": False}, pad={"t": 50}, steps=steps
        )
        layout = go.Layout(
            height=700,
            width=800,
            sliders=[sliders],
            margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
            # margin=go.layout.Margin(l=50, r=50, b=150, t=20, pad=4),
            hovermode="closest",
            template="plotly_white",
            xaxis=dict(range=[-2.2, 2.2]),
            yaxis=dict(range=[-2.2, 2.2]),
            showlegend=False,
        )
        return iplot(dict(data=data, layout=layout))

    def _tsne(self, product, config):
        config_copy = config.copy()
        for i in range(self.n_steps):
            if i == 0:
                config_copy["init"] = "pca"
            else:
                config_copy["init"] = self.plot_data["tsne"][i - 1][["x", "y"]].values
            tsne = sklearn.manifold.TSNE(**config_copy)
            X = tsne.fit_transform(self.plot_data["wi"][i])
            X = X - X.mean(axis=0)
            X = X / X.std(axis=0)
            df = product.copy()
            df["x"] = X[:, 0]
            df["y"] = X[:, 1]
            self.plot_data["tsne"].append(df)

    def _build_file_df(self, idx):
        files = [f for f in self.raw_files if re.search(r"(\d+)_(\d+).npy", f)]
        if not files:
            return None
        df = pd.DataFrame({"file": files})
        epoch_batch = df["file"].str.extract(r"(\d+)_(\d+).npy").astype(np.int32)
        epoch_batch.rename(columns={0: "epoch", 1: "batch"}, inplace=True)
        df = pd.concat([df, epoch_batch], axis=1)
        df = df.sort_values(["epoch", "batch"]).reset_index(drop=True)
        if self.n_steps < df.shape[0]:
            if idx is None:
                rows_keep = (
                    np.linspace(0, 1, self.n_steps) ** 2 * (df.shape[0] - 1)
                ).astype(int)
            else:
                # rows_keep = [i for i in idx if i <= df.shape[0]]
                rows_keep = idx
            df = df.iloc[rows_keep]
        self.file_df = df

    def _load_data(self, idx=None, agg=None):
        if len(idx) < self.n_steps:
            self.n_steps = len(idx)
        self._build_file_df(idx=idx)
        df = self.file_df
        if df is not None:
            data = []
            for i, x in df.iterrows():
                data_i = np.load(x["file"])
                if agg is not None:
                    data_i = agg(data_i)
                data.append(data_i)
            self.plot_data["wi"] = data
            self.plot_data["steps"] = [
                f"e{x:02d}-b{y:06d}" for (x, y) in zip(df.epoch.values, df.batch.values)
            ]
        self.plot_data["tsne"] = []
