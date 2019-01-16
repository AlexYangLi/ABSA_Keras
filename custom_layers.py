# -*- coding: utf-8 -*-

"""

@author: alexyang

@contact: alex.yang0326@gmail.com

@file: custom_layers.py

@time: 2019/1/5 10:02

@desc:

"""

from keras import backend as K, initializers, regularizers, constraints
from keras.engine.topology import Layer


# modified based on `https://gist.github.com/cbaziotis/7ef97ccf71cbc14366835198c09809d2`
class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
 e: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self, W_regularizer=None, u_regularizer=None, b_regularizer=None, W_constraint=None,
                 u_constraint=None, b_constraint=None, use_W=True, use_bias=False, return_self_attend=False,
                 return_attend_weight=True, **kwargs):
        self.supports_masking = True

        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.use_W = use_W
        self.use_bias = use_bias
        self.return_self_attend = return_self_attend    # whether perform self attention and return it
        self.return_attend_weight = return_attend_weight    # whether return attention weight
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        if self.use_W:
            self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),  initializer=self.init,
                                     name='{}_W'.format(self.name), regularizer=self.W_regularizer,
                                     constraint=self.W_constraint)
        if self.use_bias:
            self.b = self.add_weight(shape=(input_shape[1],), initializer='zero', name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer, constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],), initializer=self.init, name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer, constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        if self.use_W:
            x = K.tanh(K.dot(x, self.W))

        ait = Attention.dot_product(x, self.u)
        if self.use_bias:
            ait += self.b

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        if self.return_self_attend:
            attend_output = K.sum(x * K.expand_dims(a), axis=1)
            if self.return_attend_weight:
                return attend_output, a
            else:
                return attend_output
        else:
            return a

    def compute_output_shape(self, input_shape):
        if self.return_self_attend:
            if self.return_attend_weight:
                return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]
            else:
                return input_shape[0], input_shape[-1]
        else:
            return input_shape[0], input_shape[1]

    @staticmethod
    def dot_product(x, kernel):
        """
        Wrapper for dot product operation, in order to be compatible with both
        Theano and Tensorflow
        Args:
            x (): input
            kernel (): weights
        Returns:
        """
        if K.backend() == 'tensorflow':
            return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
        else:
            return K.dot(x, kernel)


class RecurrentAttention(Layer):
    """
    Multiple attentions non-linearly combined with a recurrent neural network (gru) .
    Supports Masking.
    Follows the work of Peng et al. [http://aclweb.org/anthology/D17-1047]
    "Recurrent Attention Network on Memory for Aspect Sentiment Analysis"
    """

    def __init__(self, units, n_hop=5, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.units = units
        self.n_hop = n_hop
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(RecurrentAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, list):   # input: memory(3D) & aspect(2D)
            input_mem_shape = input_shape[0]
            al_w_shape = input_shape[0][-1] + input_shape[1][-1] + self.units
        else:   # input: just memory
            input_mem_shape = input_shape
            al_w_shape = input_shape[-1] + self.units

        # attention weights
        self.al_w = self.add_weight(shape=(self.n_hop, al_w_shape, 1), initializer=self.initializer,
                                    name='{}_al_w'.format(self.name), regularizer=self.regularizer,
                                    constraint=self.constraint)
        self.al_b = self.add_weight(shape=(self.n_hop, 1), initializer='zero', name='{}_al_b'.format(self.name),
                                    regularizer=self.regularizer, constraint=self.constraint)

        # gru weights
        self.gru_wr = self.add_weight(shape=(input_mem_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wr'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_ur = self.add_weight(shape=(self.units, self.units), initializer=self.initializer,
                                      name='{}_ur'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wz = self.add_weight(shape=(input_mem_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wz'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_uz = self.add_weight(shape=(self.units, self.units), initializer=self.initializer,
                                      name='{}_uz'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wx = self.add_weight(shape=(input_mem_shape[-1], self.units), initializer=self.initializer,
                                      name='{}_wx'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        self.gru_wg = self.add_weight(shape=(self.units, self.units), initializer=self.initializer,
                                      name='{}_wg'.format(self.name), regularizer=self.regularizer,
                                      constraint=self.constraint)
        super(RecurrentAttention, self).build(input_shape)

    def call(self, inputs, mask=None):
        if isinstance(inputs, list):
            memory, aspect = inputs
            mask = mask[0]
        else:
            memory = inputs

        attend_weights = []
        batch_size = K.shape(memory)[0]
        time_steps = K.shape(memory)[1]
        e = K.zeros(shape=(batch_size, self.units))
        for h in range(self.n_hop):
            # compute attention weight
            repeat_e = K.repeat(e, time_steps)
            if isinstance(inputs, list):
                repeat_asp = K.repeat(aspect, time_steps)
                inputs_concat = K.concatenate([memory, repeat_asp, repeat_e], axis=-1)
            else:
                inputs_concat = K.concatenate([memory, repeat_e], axis=-1)
            g = K.squeeze(K.dot(inputs_concat, self.al_w[h]), axis=-1) + self.al_b[h]   # [batch_size, time_steps]
            a = K.exp(g)

            # apply mask after the exp. will be re-normalized next
            if mask is not None:
                a *= K.cast(mask, K.floatx())

            a /= K.cast(K.sum(a, axis=-1, keepdims=True) + K.epsilon(), K.floatx())
            attend_weights.append(a)

            # apply attention
            a_expand = K.expand_dims(a)    # [batch_size, time_steps, 1]
            i_AL = K.sum(memory * a_expand, axis=1)   # [batch_size, hidden], i_AL is the input of gru at time `h`

            # gru implementation
            r = K.sigmoid(K.dot(i_AL, self.gru_wr) + K.dot(e, self.gru_ur))    # reset gate
            z = K.sigmoid(K.dot(i_AL, self.gru_wz) + K.dot(e, self.gru_uz))    # update gate
            _e = K.tanh(K.dot(i_AL, self.gru_wx) + K.dot(r*e, self.gru_wg))
            e = (1 - z) * e + z * _e  # update e

        if self.return_attend_weight:
            return e, K.concatenate(attend_weights, axis=0)
        else:
            return e

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        if isinstance(input_shape, list):
            memory_shape = input_shape[0]
        else:
            memory_shape = input_shape
        if self.return_attend_weight:
            return [(memory_shape[0], self.units), (self.n_hop, memory_shape[0], memory_shape[1])]
        else:
            return memory_shape[0], self.units


class InteractiveAttention(Layer):
    """
    Interactive attention between context and aspect text.
    Supporting Masking.
    Follows the work of Dehong et al. [https://www.ijcai.org/proceedings/2017/0568.pdf]
    "Interactive Attention Networks for Aspect-Level Sentiment Classification"
    """

    def __init__(self, return_attend_weight=False, initializer='orthogonal', regularizer=None,
                 constraint=None, **kwargs):
        self.return_attend_weight = return_attend_weight

        self.initializer = initializers.get(initializer)
        self.regularizer = regularizers.get(regularizer)
        self.constraint = constraints.get(constraint)

        self.supports_masking = True
        super(InteractiveAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape

        self.context_w = self.add_weight(shape=(context_shape[-1], asp_text_shape[-1]), initializer=self.initializer,
                                         regularizer=self.regularizer, constraint=self.constraint,
                                         name='{}_context_w'.format(self.name))
        self.context_b = self.add_weight(shape=(context_shape[1],), initializer='zero', regularizer=self.regularizer,
                                         constraint=self.constraint, name='{}_context_b'.format(self.name))
        self.aspect_w = self.add_weight(shape=(asp_text_shape[-1], context_shape[-1]), initializer=self.initializer,
                                        regularizer=self.regularizer, constraint=self.constraint,
                                        name='{}_aspect_w'.format(self.name))
        self.aspect_b = self.add_weight(shape=(asp_text_shape[1],), initializer='zero', regularizer=self.regularizer,
                                        constraint=self.constraint, name='{}_aspect_b'.format(self.name))

        super(InteractiveAttention,self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        if mask is not None:
            context_mask, asp_text_mask = mask
        else:
            context_mask = None
            asp_text_mask = None

        context, asp_text = inputs

        context_avg = K.mean(context, axis=1)
        asp_text_avg = K.mean(asp_text, axis=1)

        # attention over context with aspect_text
        a_c = K.tanh(K.batch_dot(asp_text_avg, K.dot(context, self.context_w), axes=[1, 2]) + self.context_b)
        a_c = K.exp(a_c)
        if context_mask is not None:
            a_c *= K.cast(context_mask, K.floatx())
        a_c /= K.cast(K.sum(a_c, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_context = K.sum(context * K.expand_dims(a_c), axis=1)

        # attention over aspect text with context
        a_t = K.tanh(K.batch_dot(context_avg, K.dot(asp_text, self.aspect_w), axes=[1, 2]) + self.aspect_b)
        a_t = K.exp(a_t)
        if context_mask is not None:
            a_t *= K.cast(asp_text_mask, K.floatx())
        a_t = K.cast(K.sum(a_t, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        attend_asp_text = K.sum(asp_text * K.expand_dims(a_t), axis=1)

        attend_concat = K.concatenate([attend_context, attend_asp_text], axis=-1)

        if self.return_attend_weight:
            return attend_concat, a_c, a_t
        else:
            return attend_concat

    def compute_mask(self, inputs, mask=None):
        return None

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        context_shape, asp_text_shape = input_shape
        if self.return_attend_weight:
            return [(context_shape[0], context_shape[-1]+asp_text_shape[-1]), (context_shape[0], context_shape[1]),
                    (asp_text_shape[0], asp_text_shape[1])]
        else:
            return context_shape[0], context_shape[-1]+asp_text_shape[-1]
