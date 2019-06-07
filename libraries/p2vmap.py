##
##  IMPORT
##

import os                           # file path ops
import collections                  # ordered dicts
import itertools                    # build skip-gram samples from basket list
import glob                         # dashboard -- find result files in path
import re                           # regex
import json                         # saving dicts

import numpy as np                  # output data is numpy.array
import pandas as pd                 # input data is pandas.DataFrame

import tensorflow as tf             # for implementation of skip-gram in TensorFlow

import sklearn.decomposition        # PCA
import sklearn.manifold             # t-SNE

import plotly                       # plotting library
from plotly.offline import iplot    # ... more plotly stuff
import plotly.graph_objs as go      # ... more plotly stuff



##
##  Step I: Data Preparation
##

def step_1(df_train, df_validation=None, df_test=None, **kwargs):
    """
    Build data streamers and add them to the control dictionary for skip-gram model.

    Args:
      df_train: (pandas.DataFrame): training basket data
      df_validation: (pandas.DataFrame): validation basket data
      df_test: (pandas.DataFrame): testing basket data
      kwargs: (dict) arguments for preparing SG model input data, must conatain the following keys
          variable_basket: (str) variable name in data that identifies baskets
          variable_values: (str) first must be product id, other variables are center covariates (e.g., price)
          batch_size: (int) number of training samples per batch
          shuffle: (bool) should basket list be shuffled when iterator is reset?
          n_negative_samples: (int) number of negagtive samples, `n_neg` in paper
          power: (float) power of negative samples distribution, `pow` in paper

    Returns:
      data streamer for each input data set.

    For more details, see DataStreamP2V.
    """

    data_streamer_train = DataStreamP2V(data=df_train, **kwargs)

    if df_validation is not None:
        data_streamer_validation = DataStreamP2V(data=df_validation, **kwargs)
    else:
        data_streamer_validation = None

    if df_test is not None:
        data_streamer_test = DataStreamP2V(data=df_test, **kwargs)
    else:
        data_streamer_test = None

    return data_streamer_train, data_streamer_validation, data_streamer_test


class DataStreamP2V(object):
    """
    Data streamer class for skip-gram model.
    """

    def __init__(
            self,
            data,
            variable_basket='basket_hash',
            variable_values=['j', 'discount'],
            batch_size=10000,
            shuffle=True,
            n_negative_samples=0,
            power=.75,
            allow_context_collisions=False,
            verbose=0
            ):
        """
        Construct a skip-gram sample data streamer.

        Args:
          data: (pandas.DataFrame): the basket data
          variable_basket: (str) variable name in data that identifies baskets
          variable_values: (str) first must be product id, other variables are center covariates (e.g., price)
          batch_size: (int) number of training samples per batch
          shuffle: (bool) should basket list be shuffled when iterator is reset?
          n_negative_samples: (int) number of negagtive samples, `n_neg` in paper
          power: (float) power of negative samples distribution, `pow` in paper
          allow_context_collisions: (bool) allow context product to be negative sample?
          verbose: (int) if > 0, print verbose output

        Turns pandas DataFrame into basket list and resets basket iterator. Initializes negative sample
        generator if n_negative_samples > 0.
        """

        self.verbose = verbose
        self.batch_size = batch_size
        self.shuffle = shuffle

        # the first variable is the product id
        self.n_covariates = len(variable_values)-1

        # initialize the sample cache
        self.cached_samples = []

        # turn dataframe into basket list ...
        self.basket_list = self._basket_df_to_list(
                x=data,
                variable_basket=variable_basket,
                variable_values=variable_values
                )

        # ... and reset iterator
        self.reset_iterator()

        # initialize negative sample generator
        self.produce_negative_samples = n_negative_samples > 0
        if self.produce_negative_samples:
            self.allow_context_collisions = allow_context_collisions
            self.negative_samples_generator = NegativeSamplesGenerator(
                    data=data,
                    n_negative_samples=n_negative_samples,
                    batch_size=self.batch_size,
                    power=power
                    )


    def generate_batch(self):
        """
        Generate one batch, i.e., a tuple of numpy arrays: center, context, negative samples (optional),
        center covariates (optional).

        If the cache doesn't contain enough skip-gram samples, a basket is fetched from the basket iterator
        and new skip-gram samples are added to the cache.
        """

        # fill cache
        fill_cache = len(self.cached_samples) < self.batch_size
        while fill_cache:
            try:
                new_basket = next(self.basket_iterator, None)
                self.cached_samples.extend(itertools.permutations(new_basket, 2))
            except:
                fill_cache = False
            if len(self.cached_samples) >= self.batch_size:
                fill_cache = False

        # generate skip-gram pairs
        output_array = np.asarray(self.cached_samples[:self.batch_size])
        self.cached_samples = self.cached_samples[self.batch_size:]
        center = output_array[:,0,0].astype(np.int64)
        context = output_array[:,1,0].astype(np.int64)
        center_covariates = output_array[:,0,1:]

        # add negative samples
        if self.produce_negative_samples:
            if self.allow_context_collisions:
                negative_samples = self.negative_samples_generator.get_negative_samples()
            else:
                negative_samples = self.negative_samples_generator.get_negative_samples(context).T
        else:
            # build empty array
            negative_samples = output_array[:,0,10000000:] # hack: we never have that many covariate columns
        return center, context, negative_samples, center_covariates


    def reset_iterator(self):
        """
        Reset basket iterator.

        If self.shuffle is `True`, basket order is randomized.
        """

        if self.shuffle:
            np.random.shuffle(self.basket_list)

        self.basket_iterator = self._basket_iterator(self.basket_list)


    def _basket_df_to_list(self, x, variable_basket, variable_values):
        """
        (Helper method) Turn pandas basket DataFrame into basket list. This pure numpy implementation is much
        faster than alternative approaches such as df.groupby('a')['b'].apply(list).
        """

        x_basket_values = x[[variable_basket]+variable_values].sort_values([variable_basket]).values
        keys = x_basket_values[:,0]
        ukeys, index = np.unique(keys, True)
        return np.split(x_basket_values[:,1:], index)[1:]


    def _basket_iterator(self, basket_list):
        """
        (Helper method) An iterator that yields one basket at a time. Used to fill the sample cache.
        """

        for basket in basket_list:
            yield basket


class NegativeSamplesGenerator(object):
    """
    Class for generating negative samples for skip-gram model.
    """

    def __init__(self, data, n_negative_samples, batch_size, power=.75, domain=2**31-1):
        """
        Construct a negative sample generator.

        Args:
          data: (pandas.DataFrame): basket data
          n_negative_samples: (int) number of negative samples per positive sample
          batch_size: (int) number of training samples per batch
          power: (float) power of negative samples distribution, `pow` in paper
          domain: (int) integer range, used in sampling
        """

        self.counts = self._build_product_counts(data)
        self.n_negative_samples = n_negative_samples
        self.batch_size = batch_size
        self.power = power
        self.n_draws = self.batch_size * self.n_negative_samples
        self.domain = domain
        self.products = np.array(list(self.counts.keys()))
        self._build_cumulative_count_table()


    def get_negative_samples(self, context=None):
        """
        Generate negative samples.

        Args:
          context: vector of context products.

        If `context` is provided, avoid collisions between negative samples and context products.
        """

        if context is not None:
            # init
            negative_samples = np.zeros((self.n_negative_samples, len(context)), dtype=np.int32) - 1
            done_sampling = False

            # sampling is done when all values are != -1
            while not done_sampling:
                # how many samples do we need?
                new_sample_index = negative_samples == -1
                n_draws = np.sum(new_sample_index)

                # sampling integers
                random_integers = np.random.randint(0, self.domain, n_draws)

                # map integers to product ids
                new_negative_samples_index = np.searchsorted(
                        self.cumulative_count_table,
                        random_integers
                        )
                new_negative_samples = self.products[new_negative_samples_index]
                negative_samples[new_sample_index] = new_negative_samples

                # reset collisions to -1
                negative_samples[negative_samples==context] = -1

                # is sampling done?
                done_sampling = np.all(negative_samples != -1)

            return negative_samples

        else:
            # sampling integers
            random_integers = np.random.randint(0, self.domain, self.n_draws)

            # map integers to product ids
            negative_samples_index = np.searchsorted(self.cumulative_count_table, random_integers)

            # format and return
            return self.products[negative_samples_index].reshape((self.batch_size, self.n_negative_samples))


    def _build_product_counts(self, x):
        """
        (Helper method) Build dictionary that counts how often products occur in basket corpus.
        """

        # this assumes that products are integer, from 0 to df.j.max()
        n_products = x.j.max()+1
        product_counts = x.groupby('j').j.count().to_dict()

        # a product might not occur in the data. we add these to the count dict (with a 0 count).
        product_counts_filled = collections.OrderedDict()

        for j in range(n_products):
            if j not in product_counts:
                product_counts_filled[j] = 0
            else:
                product_counts_filled[j] = product_counts[j]

        product_counts_filled = product_counts_filled

        return product_counts_filled


    def _build_cumulative_count_table(self):
        """
        (Helper method) Build cumulative count table (from 0 to self.domain). This allows us to later sample
        an integer and use np.searchsorted to map this to a product.
        """

        tmp = np.array(list(self.counts.values())) ** self.power
        cumulative_relative_count_table = np.cumsum(tmp / sum(tmp))
        self.cumulative_count_table = np.int32((cumulative_relative_count_table * self.domain).round())
        assert self.cumulative_count_table[-1] == self.domain



##
##  CUSTOMIZED SG MODEL (STEP 2)
##

def step_2(**kwargs):
    """
    Train skip-gram model.

    Args:
      kwargs: (dict) arguments for training the SG model, must conatain the following keys
          p2v_kwargs: (dict) arguments for model setup, must contain the following keys
              n_products: (int) the number of products for which product vectors should be trained--this is
                the row dimension in the embedding matrices
              size: (int) the number of latent product attributes--this is the column dimension in the
                embedding matrices -- `L` in the paper
              train_streamer: a training data streamer, created by `DataStreamP2V`
              test_streamer: a test data streamer, created by `DataStreamP2V` (optional)
              validation_streamer: a validation data streamer, created by `DataStreamP2V` (optional)
              batch_size: (int) number of training samples per batch
              path_results: (str) path that results are written to
              n_batch_save: (int) save training results every n_batch_save-th step
              n_batch_validation: (int) save validation and test results every n_batch_validation-th step
              n_batch_print: (int) print training loss every n_batch_save-th step
          p2v_train_kwargs: (dict) arguments for training step, must contain the following keys
              x

    Returns:
      data streamer for each input data set.

    For more details, see DataStreamP2V.

    """

    # tensorflow session
    tf.reset_default_graph()
    tf_config_proto = tf.ConfigProto()
    tf_config_proto.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config_proto)

    # initialize model class
    p2v = Product2VecTensorflow(session=session, **kwargs.get('p2v_kwargs'))

    # train
    p2v.train(**kwargs.get('p2v_train_kwargs'))

    return p2v


## parent class
class Product2Vec(object):
    """
    Superclass for models that produce product vectors.
    """

    def __init__(
            self,
            n_products,
            size,
            train_streamer,
            test_streamer=None,
            validation_streamer=None,
            batch_size=5000,
            path_results=None,
            n_batch_save=1000,
            n_batch_validation=1000,
            n_batch_print=1000,
            verbose=0
            ):
        """
        Construct a new skip-gram model.

        Args:
          n_products: (int) the number of products for which product vectors should be trained--this is the
            row dimension in the embedding matrices
          size: (int) the number of latent product attributes--this is the column dimension in the embedding
            matrices -- `L` in the paper
          train_streamer: a training data streamer, created by `DataStreamP2V`
          test_streamer: a test data streamer, created by `DataStreamP2V` (optional)
          validation_streamer: a validation data streamer, created by `DataStreamP2V` (optional)
          batch_size: (int) number of training samples per batch
          path_results: (str) path that results are written to
          n_batch_save: (int) save training results every n_batch_save-th step
          n_batch_validation: (int) save validation and test results every n_batch_validation-th step
          n_batch_print: (int) print training loss every n_batch_save-th step
          verbose: (int) if > 0, print verbose output

        Notes:
        The number of negative samples is taken from `train_streamer` (that contains the
          `negative_samples_generator`).
        """

        self.verbose = verbose

        self.n_products = n_products
        self.size = size

        self.batch_size = batch_size

        self.n_batch_save = n_batch_save
        self.n_batch_validation = n_batch_validation
        self.n_batch_print = n_batch_print

        self.n_covariates = train_streamer.n_covariates
        self.n_negative_samples = train_streamer.negative_samples_generator.n_negative_samples

        # init counters
        self.epoch_counter = 0
        self.batch_counter = 0

        # data
        self.train_streamer = train_streamer
        self.test_streamer = test_streamer
        self.validation_streamer = validation_streamer

        # create result directory
        self.path_results = os.path.join(path_results, 'out')
        self.result_file_pattern = os.path.join(self.path_results, '%s_%d_%d.npy')
        if not os.path.exists(self.path_results):
            os.makedirs(self.path_results)


    def train(self, n_epoch, learning_rate):
        """
        This high-level method performs training over the number of requested epochs. The training data is
        batched according to the parameters of the DataStreamP2V object passed to __init__().

        Args:
          n_epoch: (int) number of passes through training data
          learning_rate: (float) optimizer learning rate
        """

        # compute learning rate(s) for all epochs
        # not used for adam-like optimizers that automatically tune learning rates
        epochs, learning_rates = self._get_epochs_learning_rates(n_epoch, learning_rate)

        # training loop over epochs
        for epoch, learning_rate_epoch in zip(epochs, learning_rates):

            # reset batch counter
            self.batch_counter = 0

            # reset training data streamer
            # it is not necessary to reset the test and validation streamer because we use the whole test and
            # validation data set(s)
            self.train_streamer.reset_iterator()

            # loss cache
            self._loss_save = []

            # update learning rate
            self._set_learning_rate(learning_rate_epoch)

            # iterate through all training examples in training data set
            while True:
                # try to generate training samples, break when data streamer is depleted
                try:
                    center, context, negative_samples, center_covariates =\
                            self.train_streamer.generate_batch()
                except:
                    break

                # update batch counter
                self.batch_counter += 1

                # do training step and current training save loss
                loss = self._training_step(
                        center,
                        context,
                        negative_samples,
                        center_covariates
                        )
                self._loss_save.append(loss)

                # store initial training loss (execute only for first batch in first epoch)
                if (self.batch_counter==1) and self.epoch_counter == 0:
                    self._save_output()

                # >> callbacks "on batch end"
                # conditioned on n_batch_print, n_batch_save and n_batch_validation
                # 1  print
                if self.batch_counter % self.n_batch_print == 0:
                    self._print_loss()
                # 2  save output for training data
                if self.batch_counter % self.n_batch_save == 0:
                    self._save_output()
                # 3  save output for test and validation data
                if self.batch_counter % self.n_batch_validation == 0:
                    self._save_output_validation_test()

            # >> callbacks "on epoch end"
            # 1  print
            self._print_loss()
            # 2  save output
            self._save_output()
            self._save_output_validation_test()
            self._save_learning_rate()

            # update epoch counter
            self._log('end of epoch %s' % self.epoch_counter)
            self.epoch_counter += 1


    def _get_epochs_learning_rates(self, n_epoch, learning_rate):
        """
        (Helper method) Convenience function for creating learning rate grids.

        Not needed for optimizers that automatically tune learning rate (e.g., ADAM).
        """

        if isinstance(learning_rate, list):
            assert len(learning_rate) == 3
            assert n_epoch % learning_rate[2] == 0
            n_level = n_epoch / learning_rate[2]
            delta_learning_rate = (learning_rate[0]-learning_rate[1])/n_level
            learning_rates = np.repeat(
                    (np.arange(learning_rate[1], learning_rate[0], delta_learning_rate) +
                        delta_learning_rate)[::-1],
                    learning_rate[2]
                    )

        else:
            learning_rates = np.repeat(learning_rate, n_epoch)

        return range(n_epoch), learning_rates


    def _print_loss(self):
        """
        (Helper method) Print average training loss.
        """

        self._log('batch = %06s  |  loss = %0.4f' % (
            str(self.batch_counter), np.mean(self._loss_save)
            ))


    def _save_output(self):
        """
        (Helper method) Save model output for training data.

        saves data:
          - loss
          - center embedding (wi) -- `v` in paper
          - context embedding (wo) -- `w` in paper
          - global bias (optional) -- `beta_0` in paper
          - product bias (optional) -- `beta_ce` and `beta_co` in paper
          - weights for covariates (optional) -- alpha
        """

        file_base = os.path.join(
                self.path_results,
                '%s_%d_%d.npy' % ('%s', self.epoch_counter, self.batch_counter)
                )

        np.save(file=file_base % 'train_loss', arr=np.array(self._loss_save))
        self._loss_save = []

        wi = self.get_wi()
        np.save(file=file_base % 'wi', arr=wi)

        wo = self.get_wo()
        np.save(file=file_base % 'wo', arr=wo)

        if self.product_bias_negative_sampling:
            bo_j, bo_j_center = self.get_bias_product()
            np.save(file_base % 'bo_j', bo_j)
            np.save(file_base % 'bo_j_center', bo_j_center)

        if self.use_covariates:
            w_ce_cov = self.get_w_ce_cov()
            np.save(file_base % 'w_ce_cov', w_ce_cov)


    def _save_output_validation_test(self):
        """
        (Helper method) Save model output for training data.

        saves data:
          - test loss
          - validation loss
        """

        if self.test_streamer is not None:

            # reset iterator in test streamer
            self.test_streamer.reset_iterator()
            self._log('compute test loss')
            test_loss_list = []

            # loop through complete test streamer
            while True:
                try:
                    center, context, negative_samples, center_covariates =\
                            self.test_streamer.generate_batch()
                except:
                    break

                # only use complete batches
                if len(center) == self.test_streamer.batch_size:
                    test_loss_i = self._session.run(
                            self._loss,
                            feed_dict=self._get_feed_dict(
                                center,
                                context,
                                negative_samples,
                                center_covariates,
                                0
                                )
                            )
                    test_loss_list.append(test_loss_i)

            # compute (and save) average test loss
            test_loss = np.mean(test_loss_list)
            np.save(
                self.result_file_pattern % ('test_loss', self.epoch_counter, self.batch_counter),
                test_loss
                )

        # TODO: "weaponize" this code (it's the same as block above for test loss)
        if self.validation_streamer is not None:

            # reset iterator in test streamer
            self.validation_streamer.reset_iterator()
            self._log('compute validation loss')
            validation_loss_list = []

            # loop through complete test streamer
            while True:
                try:
                    center, context, negative_samples, center_covariates =\
                            self.validation_streamer.generate_batch()
                except:
                    break

                # only use complete batches
                if len(center) == self.validation_streamer.batch_size:
                    validation_loss_i = self._session.run(
                            self._loss,
                            feed_dict=self._get_feed_dict(
                                center,
                                context,
                                negative_samples,
                                center_covariates,
                                0
                                )
                            )
                    validation_loss_list.append(validation_loss_i)

            # compute (and save) average validation loss
            validation_loss = np.mean(validation_loss_list)
            np.save(
                self.result_file_pattern % ('val_loss', self.epoch_counter, self.batch_counter),
                validation_loss
                )


    def _save_learning_rate(self):
        """
        (Helper method) Save learning rate.
        """

        np.save(
            self.result_file_pattern % ('learning_rate', self.epoch_counter, self.batch_counter),
            self.learning_rate_epoch
            )


    def get_wi(self):
        """
        Return center product embedding.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.

        The method is expected to return the center embedding, i.e., a numpy array of size JxL (number of
        products x number of latent dimensions).
        """

        raise NotImplementedError


    def get_wo(self):
        """
        Return context product embedding.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.

        The method is expected to return the context embedding, i.e., a numpy array of size JxL (number of
        products x number of latent dimensions).
        """

        raise NotImplementedError


    def get_bias_product(self):
        """
        Return product bias embedding.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.

        The method is expected to return the product bias embedding, i.e., a numpy array of size Jx1 (number
        of products x 1).
        """

        raise NotImplementedError


    def get_w_ce_cov(self):
        """
        Return covariate weights.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.

        The method is expected to return the covariate weights, i.e., a numpy array of size Kx1 (number of
        covariates x 1).
        """

        raise NotImplementedError



    def _training_step(self, center, context, negative_samples, center_covariates):
        """
        (Helper method) This method is responsible for updating the embedding weights based on a given batch
        of training samples.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.

        Args:
          center: center products (shape: batch_size x 1)
          context: context products (shape: batch_size x 1)
          negative_samples: negative samples (shape: batch_size x n_negative_samples)
          center_covariates: covariats for center products (shape: batch_size x n_covariates)
          return: batch_loss (or 0 if no loss is computed)
        """

        raise NotImplementedError


    def _set_learning_rate(self, learning_rate):
        """
        (Helper method) Update learning rate.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.
        """

        raise NotImplementedError


    def _forward(self, ce, co):
        """
        (Helper method) Do a forward pass, i.e., compute logits, probabilities, and ground truth arrays for
        center/context pairs.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.
        """

        raise NotImplementedError


    def _get_loss(self, truth, scores):
        """
        (Helper method) Compute the loss for given scores and truth.

        This method depends on the specific implementation (e.g., TensorFlow) so it has to be implemented in
        class that inherits from `Product2Vec`.
        """

        raise NotImplementedError


    def _log(self, x):
        """
        (Helper method) Very simple logging implementation.

        Only log if self.verbose >= 1.

        IF        self.verbose == 1   only log text
        ELSE IF   self.verbose > 1    also log other data types (e.g., np.array, pd.DataFrame, etc.)
        """

        if self.verbose > 0:
            if self.verbose > 1 or isinstance(x, str):
                print(x)


## TensorFlow model class
class Product2VecTensorflow(Product2Vec):
    """
    Skip-gram model implementation in TensorFlow, based on superclass `Product2Vec`.
    """

    def __init__(
            self,
            session,
            ws_initializer_constant=None,
            bias_negative_sampling=False,
            product_bias_negative_sampling=False,
            normalise_weights=False,
            regularisation=None,
            use_covariates=False,
            optimizer={'method':'sgd'},
            summary_ws=False,
            *args,
            **kwargs
            ):

        """
        Construct a new skip-gram model, based on TensorFlow implementation.

        Args:
          session: TensorFlow session
          ws_initializer_constant: (float) constant value for embedding initialization, use random
            initialization if `None`.
          bias_negative_sampling: (bool) use (global) output bias? -- paper: None
          product_bias_negative_sampling: (bool) use product-specific output bias? -- paper: True
          normalise_weights: (bool) normalize weights? -- paper: False
          regularisation: (bool) regularize embedding matrices? -- paper: False
          use_covariates: (bool) use covariates? -- paper: False
          optimizer: (dict) which optimizer should be used? -- paper: Adam
          summary_ws: (bool) False -- not the biggest fan of tensorboard anymore
          args: args passed to `Product2Vec.__init__`
          kwargs: kwargs passed to `Product2Vec.__init__`
        """


        # superclass constructor
        Product2Vec.__init__(self, *args, **kwargs)

        # check input: use bias_negative_sampling or for negative samples, DUH!
        if bias_negative_sampling and self.n_negative_samples==0:
            raise Exception('use bias_negative_sampling only when n_negative_samples > 0')

        # check input: either bias_negative_sampling or product_bias_negative_sampling
        if bias_negative_sampling and product_bias_negative_sampling:
            raise Exception('use either bias_negative_sampling or product_bias_negative_sampling')

        # store that we use the TensorFlow implementation for skip-gram model
        self.method = 'tensorflow'

        self._session = session
        self.ws_initializer_constant = ws_initializer_constant
        self.bias_negative_sampling = bias_negative_sampling
        self.product_bias_negative_sampling = product_bias_negative_sampling
        self.normalise_weights = normalise_weights
        self.regularisation = regularisation
        self.use_covariates = use_covariates
        self._optimizer = optimizer
        self.summary_ws = summary_ws

        # glonal counter
        self._counter_global = 0

        # build TensorFlow graph
        self._inputs()
        self._forward()
        self._optimise()

        # initialise variables
        tf.global_variables_initializer().run(session=self._session)

        # logging
        self._log('implementation = TF')
        self._log('path_results = %s' % self.path_results)
        self._log('normalise_weights = %s' % self.normalise_weights)
        self._log('covariates = %s' % self.use_covariates)
        if self.regularisation is not None:
            self._log('regularisation = %d' % self.regularisation)
        self._log('optimizer = %s' % self._optimizer['method'])


    def get_wi(self, method='np'):
        """
        Return center product embedding.

        Either as numpy.array (method='np') or as pandas.DataFrame (method='pd').
        """

        if method=='pd':
            wi = pd.DataFrame(self._session.run(self._wi), index=range(self.n_products))
        elif method=='np':
            wi = self._session.run(self._wi)
        else:
            raise NotImplementedError

        return wi


    def get_wo(self, method='np'):
        """
        Return context product embedding.

        Either as numpy.array (method='np') or as pandas.DataFrame (method='pd').
        """

        if method=='pd':
            wo = pd.DataFrame(self._session.run(self._wo), index=range(self.n_products))
        elif method=='np':
            wo = self._session.run(self._wo)
        else:
            raise NotImplementedError

        return wo


    def get_bias_product(self):
        """
        Return product bias embedding.
        """

        return self._session.run([self._bo_j, self._bo_j_center])


    def get_w_ce_cov(self):
        """
        Return covariate weights.
        """

        return self._session.run(self._w_center_covariates)


    def _inputs(self):
        """
        (Helper method) Build input placeholders.

        Note:
        Only executed once in graph creation.
        """

        self._center = tf.placeholder(dtype=tf.int32, name='center', shape=[None])
        self._context = tf.placeholder(dtype=tf.int32, name='context', shape=[None])
        self._negative_samples = tf.placeholder(
                dtype=tf.int32, name='negative_samples', shape=[None, self.n_negative_samples]
                )
        self._center_covariates = tf.placeholder(
                dtype=tf.float32, name='center_covariates', shape=[None, self.n_covariates]
                )
        self._tf_learning_rate = tf.placeholder(dtype=tf.float32, name='learning_rate', shape=[])


    def _get_feed_dict(self, center, context, negative_samples, center_covariates, learning_rate):
        """
        (Helper method) Build TensorFlow feed dictionary.
        """

        return {
                self._center: center,
                self._context: context,
                self._negative_samples: negative_samples,
                self._center_covariates: center_covariates,
                self._tf_learning_rate: learning_rate
                }


    def _forward(self):
        """
        (Helper method) Do a forward pass, i.e., compute loss, logits, and probabilities for center/context
        pairs in batch.

        Note:
        Only executed once in graph creation.
        """

        # step counter
        self._global_step = tf.Variable(0, name="global_step")

        # initializer
        if self.ws_initializer_constant is not None:
            initializer_wi = tf.constant_initializer(self.ws_initializer_constant)
            initializer_wo = tf.constant_initializer(self.ws_initializer_constant)
        else:
            initializer_wi = tf.truncated_normal_initializer(stddev=0.08)
            initializer_wo = tf.truncated_normal_initializer(stddev=0.08)

        # input weight matrix: [n_products, embedding_size]
        wi = tf.get_variable(
                name='wi',
                shape=[self.n_products, self.size],
                dtype=tf.float32,
                initializer=initializer_wi
                )
        # l2 normalization of weights
        if self.normalise_weights:
            self._debug_wi_before_l2 = wi
            wi = tf.nn.l2_normalize(wi)
        # store tensorflow variable
        self._wi = wi

        # output weight matrix: [n_products, embedding_size]. transposed.
        wo = tf.get_variable(
                name='wo',
                shape=[self.n_products, self.size],
                dtype=tf.float32,
                initializer=initializer_wo
                )
        # l2 normalization of weights
        if self.normalise_weights:
            self._debug_wo_before_l2 = wo
            wo = tf.nn.l2_normalize(wo)
        # store tensorflow variable
        self._wo = wo

        # heatmap summary variable for logging and display in tensorboard.
        # i don't like tensorboard, it's rather clumsy and doesn't allow customization of output so i created
        # a small dashboard in plotly. this code could be removed...
        if self.summary_ws:
            self._summary_wi = tf.summary.image(
                    'wi',
                    tf.reshape(tf.transpose(self._wi), [1, self.size, self.n_products, 1])
                    )
            self._summary_wo = tf.summary.image(
                    'wo',
                    tf.reshape(tf.transpose(self._wo), [1, self.size, self.n_products, 1])
                    )

        # hidden layer, i.e., representation of center product [batch_size, embedding_size]
        self._wi_center = tf.nn.embedding_lookup(self._wi, self._center)

        # negative sampling vs. softmax
        if self.n_negative_samples>0: ## >> negative sampling as proposed in paper
            # [batch_size, embedding_size]
            self._wo_positive_samples = tf.nn.embedding_lookup(self._wo, self._context)

            # [batch_size, n_negative_samples, embedding_size]
            self._wo_negative_samples = tf.nn.embedding_lookup(self._wo, self._negative_samples)

            # [batch_size]
            self._logits_positive_samples = tf.einsum('ij,ij->i', self._wi_center, self._wo_positive_samples)

            # [batch_size, nb_neg_sample]
            self._logits_negative_samples = tf.einsum('ik,ijk->ij', self._wi_center, self._wo_negative_samples)


            ## output bias -- v1: one bias value for positive and negative samples
            if self.bias_negative_sampling:
                # global output bias
                self._bo = tf.get_variable(
                        name='bo',
                        dtype=tf.float32,
                        initializer=tf.constant(-3.0, dtype=tf.float32)
                        )

                # output for debugging
                self._debug_logits_ps_before_bias = self._logits_positive_samples
                self._debug_logits_ns_before_bias = self._logits_negative_samples

                # update logits ...
                # ... for positive samples
                self._logits_positive_samples = self._logits_positive_samples + self._bo

                # ... and for  negative samples
                self._logits_negative_samples = self._logits_negative_samples + self._bo


            ## output bias -- v2: product specific bias
            if self.product_bias_negative_sampling:
                # initialize bias terms -- context
                self._bo_j = tf.get_variable(
                        name='bo_j',
                        dtype=tf.float32,
                        initializer=tf.constant(-1.5, dtype=tf.float32, shape=[self.n_products])
                        )
                # initialize bias terms -- center
                self._bo_j_center = tf.get_variable(
                        name='bo_j_center',
                        dtype=tf.float32,
                        initializer=tf.constant(-1.5, dtype=tf.float32, shape=[self.n_products])
                        )

                # look up bias terms in bias embedding, for context and center samples
                self._bo_j_positive_samples = tf.nn.embedding_lookup(self._bo_j, self._context)
                self._bo_j_negative_samples = tf.nn.embedding_lookup(self._bo_j, self._negative_samples)
                self._bo_j_center_center = tf.nn.embedding_lookup(self._bo_j_center, self._center)

                # output for debugging
                self._debug_logits_ps_before_bias = self._logits_positive_samples
                self._debug_logits_ns_before_bias = self._logits_negative_samples

                # update logits ...
                # ... for negative samples
                self._logits_positive_samples =\
                        self._logits_positive_samples +\
                        self._bo_j_positive_samples +\
                        self._bo_j_center_center

                # ... and for negative samples
                self._logits_negative_samples =\
                        self._logits_negative_samples +\
                        self._bo_j_negative_samples +\
                        tf.reshape(self._bo_j_center_center, shape=[-1,1])


            ## covariates
            if self.use_covariates:
                # initialize weights for covariates
                self._w_center_covariates = tf.get_variable(
                        name='w_center_covariates',
                        dtype=tf.float32,
                        initializer=tf.constant(0.05, dtype=tf.float32, shape=[self.n_covariates,1])
                        )

                # output for debugging
                self._debug_logits_ps_before_covariates = self._logits_positive_samples
                self._debug_logits_ns_before_covariates = self._logits_negative_samples

                # update logits ...
                # ... for positive samples
                self._logits_positive_samples =\
                        self._logits_positive_samples +\
                        tf.squeeze(tf.matmul(self._center_covariates, self._w_center_covariates))

                # ... and for negative samples
                self._logits_negative_samples = \
                        self._logits_negative_samples +\
                        tf.matmul(self._center_covariates, self._w_center_covariates)


            ## loss
            # crossentropy = loss for positive (context) samples
            crossentropy_true = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(self._logits_positive_samples),
                    logits=self._logits_positive_samples
                    )

            # crossentropy = loss for negative (context) samples
            crossentropy_sampled = tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(self._logits_negative_samples),
                    logits=self._logits_negative_samples
                    )

            # total (average) batch loss
            loss = tf.reduce_mean(
                    # division yields average per batch row
                    # average over (positive and negative) samples [Bx1]
                    tf.divide(
                        # sum all losses
                        crossentropy_true + tf.reduce_sum(crossentropy_sampled, axis=1),
                        (self.n_negative_samples + 1)
                        )
                    )

        else: ## softmax -- don't use this, just for tests
            # [batch_size, n_products]
            self._logits = tf.matmul(self._wi_center, tf.transpose(self._wo))

            # [batch_size, n_products]
            #self._probabilities = tf.nn.softmax(self._logits)

            loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self._context,
                        logits=self._logits,
                        name='loss'
                        )
                    )

        # add regularisation
        if self.regularisation is not None:
            self._debug_loss_before_l2 = loss
            loss += self.regularisation * tf.nn.l2_loss(wi)
            loss += self.regularisation * tf.nn.l2_loss(wo)

        # save loss
        self._loss = loss

        # save loss for tensorboard... you know what i'm going to say next...
        self._summary_loss = tf.summary.scalar("loss", self._loss)
        self._summary_learning_rate = tf.summary.scalar("learning_rate", self._tf_learning_rate)
        if self.bias_negative_sampling:
            self._summary_bias = tf.summary.scalar("bias", self._bo)


    def _set_learning_rate(self, learning_rate):
        """
        (Helper method) Update learning rate.
        """

        self.learning_rate_epoch = learning_rate


    def _optimise(self):
        """
        (Helper method) Build the training operator, used to minimize loss.

        Supported are
          - (plain vanilla) stochastic gradient descent
          - stochastic gradient descent with momentum
          - Adagrad
          - RMSprop
          - >> Adam << (paper)

        Note:
        Only executed once in graph creation.
        """

        if self._optimizer['method']=='sgd':
            optimizer = tf.train.GradientDescentOptimizer(self._tf_learning_rate)

        elif self._optimizer['method']=='momentum':
            optimizer = tf.train.MomentumOptimizer(self._tf_learning_rate, **self._optimizer['control'])

        elif self._optimizer['method']=='adagrad':
            # keras documentation
            # >>> Adagrad is an optimizer with parameter-specific learning rates, which are adapted relative
            # to how frequently a parameter gets updated during training. The more updates a parameter
            # receives, the smaller the updates. It is recommended to leave the parameters of this optimizer
            # at their default values.
            #
            # keras.optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
            # tf.train.AdagradOptimizer.__init__(learning_rate, initial_accumulator_value=0.1)
            optimizer = tf.train.AdagradOptimizer(self._tf_learning_rate, **self._optimizer['control'])

        elif self._optimizer['method']=='adadelta':
            # keras documentation
            # >>> Adadelta is a more robust extension of Adagrad that adapts learning rates based on a moving
            # window of gradient updates, instead of accumulating all past gradients. This way, Adadelta
            # continues learning even when many updates have been done. Compared to Adagrad, in the original
            # version of Adadelta you don't have to set an initial learning rate. In this version, initial
            # learning rate and decay factor can be set, as in most other Keras optimizers. It is recommended
            # to leave the parameters of this optimizer at their default values.
            #
            # keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
            # tf.train.AdadeltaOptimizer.__init__(learning_rate=0.001, rho=0.95, epsilon=1e-08)
            optimizer = tf.train.AdadeltaOptimizer(self._tf_learning_rate, **self._optimizer['control'])

        elif self._optimizer['method']=='rmsprop':
            # keras documentation
            # >>> It is recommended to leave the parameters of this optimizer at their default values (except
            # the learning rate, which can be freely tuned). This optimizer is usually a good choice for
            # recurrent neural networks.
            #
            # keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
            # tf.train.RMSPropOptimizer.__init__( learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10)
            optimizer = tf.train.RMSPropOptimizer(self._tf_learning_rate, **self._optimizer['control'])

        elif self._optimizer['method']=='adam':
            # keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
            # tf.train.AdamOptimizer__init__(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)
            optimizer = tf.train.AdamOptimizer(self._tf_learning_rate, **self._optimizer['control'])

        else:
            raise NotImplementedError

        # train operator
        self._train = optimizer.minimize(self._loss, global_step=self._global_step)


    def _training_step(self, center, context, negative_samples, center_covariates):
        """
        (Helper method) This method is responsible for updating the embedding weights based on a given batch
        of training samples.

        Args:
          center: center products (shape: batch_size x 1)
          context: context products (shape: batch_size x 1)
          negative_samples: negative samples (shape: batch_size x n_negative_samples)
          center_covariates: covariates for center products (shape: batch_size x n_covariates)
          return: batch_loss (or 0 if no loss is computed)
        """

        loss, _ = self._session.run(
                [self._loss, self._train],
                feed_dict=self._get_feed_dict(
                    center,
                    context,
                    negative_samples,
                    center_covariates,
                    self.learning_rate_epoch)
                )

        self._counter_global += 1

        return loss



##
##  MAPPING (STEP 3)
##


def step_3(master, **kwargs):
    """
    (fix me)

    Wrapper for t-SNE (step 3 of P2V-MAP in paper).

    Five sequential steps:
      1. load product embedding
      2. L2 normalization
      3. PCA step for pre-reduction of product embedding (optional), only for large embeddings (L>>30)
      4. t-SNE step (maaten 2014)
      5. plot

    Args:
      master: (pandas.DataFrame) product meta data, e.g., true categories in the case of the simulation
      kwargs: (dict) arguments for mapping step, must conatain the following keys
          tsne_data_kwargs: (dict) arguments for data pre-processing, must conatain the following keys
              epoch: (int) counter of used sg model epoch
              batch: (int) counter of used sg model batch
              l2norm: (bool) should l2-norm be applied to product embedding (typically good for L>10)
              pca: (dict) arguments for sklearn.decomposition.PCA (excluding `random_state`!)
              path_results: (str) path that contains p2v results
          tsne_kwargs: (dict) arguments t-SNE step, must conatain the following keys
              random_state: (int) seed for random number generator
              n_components: (int) number of latent dimensions in reduced sapce, should be 2
              n_iter: (int) maximum number of iterations
              perplexity: (int) t-SNE perplexity
              init: (str or numpy.array) either 'pca' (to use PCA step to initalize map coordinates (`y` in
                paper) or initial map coordinates as numpy array
              angle: (float) theta parameter in maaten 2014, should be 0.5
      config_model: (dict) arguments for sklearn.manifold.TSNE (including `random_state`)

    Note:
    Using `random_state` ensures reproducibility. For simplicity, we use the same random state for PCA and
    t-SNE.
    """

    tsne_data_kwargs = kwargs.get('tsne_data_kwargs')
    tsne_kwargs = kwargs.get('tsne_kwargs')

    # 1. load product embedding
    x = np.load('{path_results}/out/wi_{epoch}_{batch}.npy'.format(**tsne_data_kwargs))

    # 2. L2 normalization
    if tsne_data_kwargs['l2norm']:
        x /= np.linalg.norm(x, axis=1)[:,np.newaxis]

    # 3. PCA step for pre-reduction of product embedding (optional), only do for large embeddings (L>>30)
    if tsne_data_kwargs['pca'] is not None:
        x = sklearn.decomposition.PCA(random_state=config_model['random_state'], **tsne_data_kwargs['pca']).fit_transform(x)

    # 4. t-SNE step
    res_tsne = sklearn.manifold.TSNE(**tsne_kwargs).fit_transform(x)

    # 5. plot
    tsne_map_xy = pd.DataFrame(
        res_tsne,
        index=master[['c', 'j']].set_index(['c','j']).index, columns=['x','y']
    ).reset_index()

    plot_map(tsne_map_xy)

    return tsne_map_xy


def plot_map(data):
    """
    Plot product map.

    Args:
      data: (pd.DataFrame) the product map data, must contain
          x: (float) x coordinate
          y: (float) y coordinate
          c: (int) category id
          j: (int) product id
    """

    dt = data.reset_index()

    # map scatter plot trace
    # use categories (c) as bubble color
    plot_data = [
        go.Scatter(
            x=dt['x'].values,
            y=dt['y'].values,
            text=[
                'category = %d <br> product = %d' % (x, y)
                for (x, y) in zip(dt['c'].values, dt['j'].values)
                ],
            hoverinfo='text',
            mode='markers',
            marker=dict(
                size=14,
                color=dt['c'].values,
                colorscale='Jet',
                showscale=False
            )
        )
    ]

    # customize plot layout
    plot_layout = go.Layout(
        width=800,
        height=600,
        margin=go.layout.Margin(l=0, r=0, b=0, t=0, pad=4),
        hovermode='closest'
    )

    # plot
    iplot(go.Figure(data=plot_data, layout=plot_layout))


