import glob
import re
import numpy as np
import pandas as pd

import sklearn.cluster              # k-means
import sklearn.metrics              # metrics: NMIC, AMIC, SIL

import scipy.spatial.distance       # compute Euclidean distances (method cdist)

import plotly                       # awesome plotting library
from plotly.offline import iplot    # ... more plotly stuff
import plotly.graph_objs as go      # ... more plotly stuff



##
##  DASHBOARD
##

class DashboardTensorFlowSG(object):
    """
    Class for visualizing skip-gram model results.

    Currently supported plots:
      - loss (training, test, validation)
      - embedding matrices (center, context)
      - price covariate
    """

    def __init__(
            self,
            path,
            master,
            n_lineplot=None,
            n_heatmap=20,
            file_loss_train='train_loss_*.npy',
            file_loss_val='val_loss_*.npy',
            file_loss_test='test_loss_*.npy',
            file_price='w_ce_cov_*.npy',
            file_wi='wi_*.npy',
            file_wo='wo_*.npy'
            ):
        """
        Constructor for dashboard instance

        Args:
          path: (str) the (absolute) path containing the skip-gram model results
          master: (pandas.DataFrame) master data, must contain columns `j` (product id) and `c` (category id)
          n_lineplot: (int) number of points used in loss line plot plotly output
          n_heatmap: (int) number of embedding heat maps displayed in plotly output
          file_loss_train: (str) file pattern identifying training loss files
          file_loss_val: (str) file pattern identifying validation loss files
          file_loss_test: (str) file pattern identifying test loss files
          file_price: (str) file pattern identifying price weight files
          file_wi: (str) file pattern identifying center embedding files
          file_wo: (str) file pattern identifying context embedding files
        """

        self.path = path
        self.master = master
        self.n_lineplot = n_lineplot
        self.n_heatmap = n_heatmap

        # store file patterns in dictionary
        self.files = {
            'file_loss_train': file_loss_train,
            'file_loss_val': file_loss_val,
            'file_loss_test': file_loss_test,
            'file_price': file_price,
            'file_wi': file_wi,
            'file_wo': file_wo,
        }

        # are validation loss results available?
        self.validation = self._build_file_df(self.files['file_loss_val'], None) is not None

        # are test loss results available?
        self.test = self._build_file_df(self.files['file_loss_test'], None) is not None

        # init plot data dictionary
        self.plot_data = {}


    def plot_loss(self, reload=True):
        """
        Create loss (line) plots.
        """

        # reload data
        if reload:
            self._load_data(
                    file=self.files['file_loss_train'],
                    label='train_loss',
                    n_plots=self.n_lineplot,
                    agg=np.mean
                    )

            if self.validation:
                self._load_data(
                        file=self.files['file_loss_val'],
                        label='val_loss',
                        n_plots=self.n_lineplot,
                        agg=np.mean
                        )

            if self.test:
                self._load_data(
                        file=self.files['file_loss_test'],
                        label='test_loss',
                        n_plots=self.n_lineplot,
                        agg=np.mean
                        )

        # training loss trace
        plot_data = [
            go.Scatter(
                x=self.plot_data['train_loss']['steps'],
                y=self.plot_data['train_loss']['trace'],
                name='train'
                )
        ]

        # add validation loss trace
        if self.validation:
            plot_data.append(
                go.Scatter(
                    x=self.plot_data['val_loss']['steps'],
                    y=self.plot_data['val_loss']['trace'],
                    name='validation'
                    )
            )

        # add test loss trace
        if self.test:
            plot_data.append(
                go.Scatter(
                    x=self.plot_data['test_loss']['steps'],
                    y=self.plot_data['test_loss']['trace'],
                    name='test'
                    )
            )

        # customize plot layout
        plot_layout = go.Layout(
            width=1200,
            height=400,
            autosize=False,
            margin=go.layout.Margin(l=50, r=50, b=100, t=0, pad=4),
            legend=dict(x=.5, y=1, xanchor='right', yanchor='top')
        )

        # plot
        iplot(go.Figure(data=plot_data, layout=plot_layout))


    def plot_price_covariate(self, reload=True):
        """
        Create price covariate (line) plot.
        """

        # reload data
        if reload:
            self._load_data(
                    file=self.files['file_price'],
                    label='price',
                    n_plots=None,
                    agg=None,
                    merge_master=False
                    )

        # price covariate weight trace
        plot_data = [
                go.Scatter(
                    x=self.plot_data['price']['steps'],
                    y=[x[0][0] for x in self.plot_data['price']['trace']],
                    name='price'
                    )
                ]

        # customize plot layout
        plot_layout = go.Layout(
                width=1200,
                height=400,
                autosize=False,
                margin=go.layout.Margin(l=50, r=50, b=100, t=0, pad=4)
                )

        # plot
        iplot(go.Figure(data=plot_data, layout=plot_layout))


    def plot_product_embedding(
            self,
            label,
            size=None,
            l2norm=True,
            transpose=True,
            zmin=-.5,
            zmax=.5,
            reload=True,
            merge_master=True
            ):
        """
        Create product embedding (heat map) plot.

        Args:
          label: (str) use `label='wi'` for center embedding and `label='wo'` for contexts embedding
          size: (int) reshape raw data to `size` rows?
          l2norm: (bool) L2 normalization of raw embedding? -- paper: True
          transpose: (bool) transpose embedding? True is good, LxJ is better to dispay than JxL
          zmin: (float) minimum z value (color scale), .5 is a good value
          zmax: (float) maximum z value (color scale), .5 is a good value
          reload: (bool) reload data?
          merge_master: (bool) merge master? sorts embedding by category `c`, good for visualization

        Note:
        This method is "porcelain", `_plot_heatmap` is "plumbing"...
        """

        # reload data
        if reload:
            self._load_data(
                    file=self.files[label],
                    label=label,
                    n_plots=self.n_heatmap,
                    merge_master=merge_master
                    )

        data_zip = zip(self.plot_data[label]['steps'], self.plot_data[label]['trace'])
        n = len(self.plot_data[label]['steps'])

        # plumbing
        self._plot_heatmap(data_zip, n, size, l2norm, transpose, zmin, zmax)


    def _plot_heatmap(self, x, n, size, l2norm, transpose, zmin, zmax):
        """
        (Helper method) Plot heat map

        ... the plumbing.
        """

        # init data and step (for sliders)
        data = []
        steps = []

        # loop over slider steps
        for i, (step_i, data_i) in enumerate(x):

            # data (and data formatting)
            if size is not None:
                data_i = data_i.reshape(size)
            if l2norm:
                data_i /= np.linalg.norm(data_i, axis=1)[:,np.newaxis]
            if transpose:
                data_i = data_i.T
            data.append(go.Heatmap(z=data_i, colorscale='Jet', zmin=zmin, zmax=zmax))

            # step
            step = dict(
                method='restyle',
                label=step_i,
                args=['visible', [False] * n],
            )
            step['args'][1][i] = True
            steps.append(step)

        # build slider
        sliders = dict(
            active=0,
            currentvalue={'visible': False},
            pad={'t': 50},
            steps=steps
        )

        # customize layout
        layout = dict(
            height=600,
            sliders=[sliders],
            margin=go.layout.Margin(l=50, r=50, b=150, t=20, pad=4)
        )

        # plot
        iplot(dict(data=data, layout=layout))


    def _build_file_df(self, file, n_plots):
        """
        (Helper method) Build dataframe with result files to be used in visualization.
        """

        # match result files
        files = [f for f in glob.glob('%s/%s' % (self.path, file)) if re.search(r'(\d+)_(\d+).npy', f)]
        if not files:
            return None

        # build file overview dataframe
        df = pd.DataFrame({'file': files})
        epoch_batch = df['file'].str.extract(r'(\d+)_(\d+).npy').astype(np.int32)
        epoch_batch.rename(columns={0: 'epoch', 1: 'batch'}, inplace=True)
        df = pd.concat([df, epoch_batch], axis=1)

        # sort by epoch and batch, prune to `n_plots`
        df = df.sort_values(['epoch', 'batch']).reset_index(drop=True)
        if n_plots is not None:
            if n_plots < df.shape[0]:
                rows_keep = np.linspace(start=0, stop=(df.shape[0]-1), num=n_plots).astype(np.int32)
                df = df.iloc[rows_keep]
        return df


    def _load_data(self, file, label, n_plots, agg=None, merge_master=False):
        """
        (Helper method) Load data for plot.

        This method is used for loss files (scalars) and embedding files (numpy arrays).
        """

        # build dataframe with result files to be used in plot
        df = self._build_file_df(file, n_plots)

        # load data for all (used) files
        if df is not None:
            data = []
            for i, x in df.iterrows():
                data_i = np.load(x['file'])
                if merge_master:
                    data_i = pd.DataFrame(data_i)
                    data_i['j'] = range(data_i.shape[0])
                    data_i = data_i.merge(self.master, on='j')
                    data_i = data_i.sort_values(['c', 'j']).set_index(['c', 'j']).values

                # aggregation, e.g., mean for loss
                if agg is not None:
                    data_i = agg(data_i)

                data.append(data_i)

            # build data dictionary required by plotly
            self.plot_data[label] = {
                'trace': data,
                'steps': ['e%d-b%dk' % (x, y) for (x, y) in zip(df.epoch.values, df.batch.values)]
            }
        else:
            self.plot_data[label] = None



##
##  BENCHMARKING
##

def benchmarking(x):
    """
    Compute scores for venchmarking metrics.

    Args:
      x: (pd.DataFrame) the product map data, must contain
          x: (float) x coordinate
          y: (float) y coordinate
          c: (int) category id
          j: (int) product id

    Note:
    This method assumes that all categories have the same size.
    """

    n_j_by_c = x.groupby('c')[['j']].nunique()
    assert n_j_by_c.j.nunique() == 1
    J_c = n_j_by_c.j.values[0]

    df_metrics = x.set_index(['j', 'c'])

    true_clusters = df_metrics.reset_index()['c'].values

    kmeans = sklearn.cluster.KMeans(n_clusters=len(np.unique(true_clusters)), n_init=30)
    predicted_clusters = kmeans.fit_predict(x.values)

    # Equation XXX --6
    silhouette_score = sklearn.metrics.silhouette_score(X=df_metrics.values, labels=true_clusters)

    # Equation
    #normalized_mutual_info_score = sklearn.metrics.normalized_mutual_info_score(
    #        labels_true=true_clusters,
    #        labels_pred=predicted_clusters
    #        )

    # Equation XXX --7
    adjusted_mutual_info_score = sklearn.metrics.adjusted_mutual_info_score(
            labels_true=true_clusters,
            labels_pred=predicted_clusters,
            average_method='arithmetic'
            )

    # Equation XXX --8
    nn_hitrate = get_hitrate(df_metrics.reset_index(), J_c-1)

    print('silhouette_score = %04f' % silhouette_score)
    #print('normalized_mutual_info_score = %04f' % normalized_mutual_info_score)
    print('adjusted_mutual_info_score = %04f' % adjusted_mutual_info_score)
    print('nn_hitrate = %0.4f' % nn_hitrate)

    return silhouette_score, adjusted_mutual_info_score, nn_hitrate


def get_hitrate(x, n):
    """
    Compute NN benchmarking metric.

    Args:
      x: (pd.DataFrame) the product map data, must contain
          x: (float) x coordinate
          y: (float) y coordinate
          c: (int) category id
          j: (int) product id
      n: (int) the number of nearest neighbors (per category) that should be in the same product cluster as
        the reference product. Is equals the number of products per category minus 1, `J_c-1`

    Note:
    This method assumes that all categories have the same size.
    """

    # extract values from input data
    xys = x[['x','y']].values
    js = x['j'].values
    cs = x['c'].values

    # compute Euclidean distances between all products
    distance_in_plot = scipy.spatial.distance.cdist(xys, xys)

    # build distance DataFrame
    distance_df = pd.DataFrame({
        'j': np.repeat(js, len(js)),
        'c': np.repeat(cs, len(cs)),
        'j2': np.tile(js, len(js)),
        'c2': np.tile(cs, len(cs)),
        'd': distance_in_plot.flatten()
    })
    assert distance_df[['j', 'c']].drop_duplicates().shape[0] == len(js)
    assert distance_df[['j2', 'c2']].drop_duplicates().shape[0] == len(js)

    # remove cases where j == j2 ("self-distance", which is 0)
    distance_df = distance_df[distance_df['j'] != distance_df['j2']]

    # prune data to nearest neighbors
    distance_df = distance_df.sort_values('d')
    distance_df['rank_d'] = distance_df.groupby('j').cumcount()
    nn = distance_df[distance_df['rank_d']<n]

    # calculate score: fraction of nearest neighbors for which category is identical to reference product
    score = float(sum(nn['c'] == nn['c2'])) / nn.shape[0]
    return score


