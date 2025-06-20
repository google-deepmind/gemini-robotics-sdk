# Copyright 2025 Google LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Interface for TF-Agents models, indepdent of the package.

Based on tf_agents v0.19.0.

This is a temporary solution to allow us to use TF-Agents models in Safari
without depending on TF-Agents directly. TF-Agents has a lot of dependencies
with strict version requirements, and is only lightly maintained, making it a
source of many dependency conflicts.
"""

import numbers
import pprint
from typing import Any, Iterable, Mapping, NamedTuple, Optional, Sequence, Text, Type, TypeVar, Union

import numpy as np
import tensorflow as tf


# from tf_agents.distributions.utils
class Params(object):
  """The (recursive) parameters of objects exposing the `parameters` property.

  This includes TFP `Distribution`, `Bijector`, and TF `LinearOperator`.

  `Params` objects are created with
  `tf_agents.distributions.utils.get_parameters`;
  `Params` can be converted back to original objects via
  `tf_agents.distributions.utils.make_from_parameters`.

  In-place edits of fields are allowed, and will not modify the original
  objects (with the exception of, e.g., reference objects like `tf.Variable`
  being modified in-place).

  The components of a `Params` object are: `type_` and `params`.

  - `type_` is the type of object.
  - `params` is a `dict` of the (non-default) non-tensor arguments passed to the
    object's `__init__`; and includes nests of Python objects, as well as other
    `Params` values representing "Param-representable" objects passed to init.

  A non-trivial example:

  ```python
  scale_matrix = tf.Variable([[1.0, 2.0], [-1.0, 0.0]])
  d = tfp.distributions.MultivariateNormalDiag(
      loc=[1.0, 1.0], scale_diag=[2.0, 3.0], validate_args=True)
  b = tfp.bijectors.ScaleMatvecLinearOperator(
      scale=tf.linalg.LinearOperatorFullMatrix(matrix=scale_matrix),
      adjoint=True)
  b_d = b(d)
  p = utils.get_parameters(b_d)
  ```

  Then `p` is:

  ```python
  Params(
      tfp.distributions.TransformedDistribution,
      params={
          "bijector": Params(
              tfp.bijectors.ScaleMatvecLinearOperator,
              params={"adjoint": True,
                      "scale": Params(
                          tf.linalg.LinearOperatorFullMatrix,
                          params={"matrix": scale_matrix})}),
          "distribution": Params(
              tfp.distributions.MultivariateNormalDiag,
              params={"validate_args": True,
                      "scale_diag": [2.0, 3.0],
                      "loc": [1.0, 1.0]})})
  ```

  This structure can be manipulated and/or converted back to a `Distribution`
  instance via `make_from_parameters`:

  ```python
  p.params["distribution"].params["loc"] = [0.0, 0.0]

  # The distribution `new_b_d` will be a MVN centered on `(0, 0)` passed through
  # the `ScaleMatvecLinearOperator` bijector.
  new_b_d = utils.make_from_parameters(p)
  ```
  """

  type_: Type[Any]  # Any class that has a .parameters.
  params: Mapping[Text, Any]

  def __str__(self):
    return '<Params: type={}, params={}>'.format(self.type_, self.params)

  def __repr__(self):
    return str(self)

  def __eq__(self, other):
    return (
        isinstance(self, type(other))
        and self.type_ == other.type_
        and self.params == other.params
    )

  def __init__(self, type_, params):
    self.type_ = type_
    self.params = params


def _check_no_tensors(parameters: Params):
  flat_params = tf.nest.flatten(parameters.params)
  for p in flat_params:
    if isinstance(p, Params):
      _check_no_tensors(p)
    if tf.is_tensor(p):
      raise TypeError(
          'Saw a `Tensor` value in parameters:\n  {}'.format(parameters)
      )


class DistributionSpecV2(object):
  """Describes a tfp.distribution.Distribution using nested parameters."""

  def __init__(
      self, event_shape: tf.TensorShape, dtype: tf.DType, parameters: Params
  ):
    """Construct a `DistributionSpecV2` from a Distribution's properties.

    Note that the `parameters` used to create the spec should contain
    `tf.TypeSpec` objects instead of tensors.  We check for this.

    Args:
      event_shape: The distribution's `event_shape`.  This is the shape that
        `distribution.sample()` returns.  `distribution.sample(sample_shape)`
        returns tensors of shape `sample_shape + event_shape`.
      dtype: The distribution's `dtype`.
      parameters: The recursive parameters of the distribution, with tensors
        having directly been converted to `tf.TypeSpec` objects.

    Raises:
      TypeError: If for any entry `x` in `parameters`: `tf.is_tensor(x)`.
    """
    _check_no_tensors(parameters)
    self._event_shape = event_shape
    self._dtype = dtype
    self._parameters = parameters
    self._event_spec = tf.TensorSpec(shape=event_shape, dtype=dtype)

  @property
  def event_shape(self) -> tf.TensorShape:
    return self._event_shape

  @property
  def dtype(self) -> tf.DType:
    return self._dtype

  @property
  def event_spec(self) -> tf.TensorSpec:
    return self._event_spec

  @property
  def parameters(self) -> Params:
    return self._parameters

  def __eq__(self, other):
    return (
        isinstance(self, type(other))
        and self._event_shape == other._event_shape
        and self._dtype == other._dtype
        and self._parameters == other._parameters
    )

  def __str__(self):
    return (
        '<DistributionSpecV2: event_shape={}, dtype={}, parameters={}>'.format(
            self.event_shape, self.dtype, self.parameters
        )
    )

  def __repr__(self):
    return str(self)


# from tf_agents.specs.array_spec
class ArraySpec(object):
  """Describes a numpy array or scalar shape and dtype.

  An `ArraySpec` allows an API to describe the arrays that it accepts or
  returns, before that array exists.
  The equivalent version describing a `tf.Tensor` is `TensorSpec`.
  """

  __hash__ = None
  __slots__ = ('_shape', '_dtype', '_name')

  def __init__(self, shape, dtype, name=None):
    """Initializes a new `ArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    self._shape = tuple(shape)
    self._dtype = np.dtype(dtype)
    self._name = name

  @property
  def shape(self):
    """Returns a `tuple` specifying the array shape."""
    return self._shape

  @property
  def dtype(self):
    """Returns a numpy dtype specifying the array dtype."""
    return self._dtype

  @property
  def name(self):
    """Returns the name of the ArraySpec."""
    return self._name

  def __repr__(self):
    return 'ArraySpec(shape={}, dtype={}, name={})'.format(
        self.shape, repr(self.dtype), repr(self.name)
    )

  def __eq__(self, other):
    """Checks if the shape and dtype of two specs are equal."""
    if not isinstance(other, ArraySpec):
      return False
    return self.shape == other.shape and self.dtype == other.dtype

  def __ne__(self, other):
    return not self == other

  def check_array(self, array):
    """Return whether the given NumPy array conforms to the spec.

    Args:
      array: A NumPy array or a scalar. Tuples and lists will not be converted
        to a NumPy array automatically; they will cause this function to return
        false, even if a conversion to a conforming array is trivial.

    Returns:
      True if the array conforms to the spec, False otherwise.
    """
    if isinstance(array, np.ndarray):
      return self.shape == array.shape and self.dtype == array.dtype
    elif isinstance(array, numbers.Number):
      return self.shape == tuple() and self.dtype == np.dtype(type(array))
    else:
      return False

  def __reduce__(self):
    return (ArraySpec, (self.shape, self.dtype, self.name))

  @staticmethod
  def from_array(array, name=None):
    """Construct a spec from the given array or number."""
    if isinstance(array, np.ndarray):
      return ArraySpec(array.shape, array.dtype, name)
    elif isinstance(array, numbers.Number):
      return ArraySpec(tuple(), type(array), name)
    else:
      raise ValueError('Array must be a np.ndarray or number. Got %r.' % array)

  @staticmethod
  def from_spec(spec):
    """Construct a spec from the given spec."""
    return ArraySpec(spec.shape, spec.dtype, spec.name)

  def replace(self, shape=None, dtype=None, name=None):
    shape = self.shape if shape is None else shape
    dtype = self.dtype if dtype is None else dtype
    name = self.name if name is None else name
    return ArraySpec(shape, dtype, name)


def is_discrete(spec):
  return issubclass(np.dtype(spec).type, np.integer)


class BoundedArraySpec(ArraySpec):
  """An `ArraySpec` that specifies minimum and maximum values.

  Example usage:
  ```python
  # Specifying the same minimum and maximum for every element.
  spec = BoundedArraySpec((3, 4), np.float64, minimum=0.0, maximum=1.0)

  # Specifying a different minimum and maximum for each element.
  spec = BoundedArraySpec(
      (2,), np.float64, minimum=[0.1, 0.2], maximum=[0.9, 0.9])

  # Specifying the same minimum and a different maximum for each element.
  spec = BoundedArraySpec(
      (3,), np.float64, minimum=-10.0, maximum=[4.0, 5.0, 3.0])
  ```

  Bounds are meant to be inclusive. This is especially important for
  integer types. The following spec will be satisfied by arrays
  with values in the set {0, 1, 2}:
  ```python
  spec = BoundedArraySpec((3, 4), np.int, minimum=0, maximum=2)
  ```
  """

  __hash__ = None
  __slots__ = ('_minimum', '_maximum')

  def __init__(self, shape, dtype, minimum=None, maximum=None, name=None):
    """Initializes a new `BoundedArraySpec`.

    Args:
      shape: An iterable specifying the array shape.
      dtype: numpy dtype or string specifying the array dtype.
      minimum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      maximum: Number or sequence specifying the maximum element bounds
        (inclusive). Must be broadcastable to `shape`.
      name: Optional string containing a semantic name for the corresponding
        array. Defaults to `None`.

    Raises:
      ValueError: If `minimum` or `maximum` are not broadcastable to `shape` or
        if the limits are outside of the range of the specified dtype.
      TypeError: If the shape is not an iterable or if the `dtype` is an invalid
        numpy dtype.
    """
    super(BoundedArraySpec, self).__init__(shape, dtype, name)

    try:
      np.broadcast_to(minimum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError(  # pylint: disable=raise-missing-from
          'minimum is not compatible with shape. Message: {!r}.'.format(
              numpy_exception
          )
      )

    try:
      np.broadcast_to(maximum, shape=shape)
    except ValueError as numpy_exception:
      raise ValueError(  # pylint: disable=raise-missing-from
          'maximum is not compatible with shape. Message: {!r}.'.format(
              numpy_exception
          )
      )

    tf_dtype = tf.as_dtype(self._dtype)
    low = tf_dtype.min
    high = tf_dtype.max

    if minimum is None:
      minimum = low
    if maximum is None:
      maximum = high

    self._minimum = np.array(minimum)
    self._maximum = np.array(maximum)

    if tf_dtype.is_floating:
      # Replacing infinities with extreme finite float values.
      self._minimum[self._minimum == -np.inf] = low
      self._minimum[self._minimum == np.inf] = high

      self._maximum[self._maximum == -np.inf] = low
      self._maximum[self._maximum == np.inf] = high

    if np.any(self._minimum > self._maximum):
      raise ValueError(
          'Spec bounds min has values greater than max: [{},{}]'.format(
              self._minimum, self._maximum
          )
      )
    if (
        np.any(self._minimum < low)
        or np.any(self._minimum > high)
        or np.any(self._maximum < low)
        or np.any(self._maximum > high)
    ):
      raise ValueError(
          'Spec bounds [{},{}] not within the range [{}, {}] of the given '
          'dtype ({})'.format(
              self._minimum, self._maximum, low, high, self._dtype
          )
      )

    self._minimum = self._minimum.astype(self._dtype)
    self._minimum.setflags(write=False)

    self._maximum = self._maximum.astype(self._dtype)
    self._maximum.setflags(write=False)

  @classmethod
  def from_spec(cls, spec, name=None):
    if name is None:
      name = spec.name

    if hasattr(spec, 'minimum') and hasattr(spec, 'maximum'):
      return BoundedArraySpec(
          spec.shape, spec.dtype, spec.minimum, spec.maximum, name
      )

    return BoundedArraySpec(spec.shape, spec.dtype, name=name)

  @property
  def minimum(self):
    """Returns a NumPy array specifying the minimum bounds (inclusive)."""
    return self._minimum

  @property
  def maximum(self):
    """Returns a NumPy array specifying the maximum bounds (inclusive)."""
    return self._maximum

  @property
  def num_values(self):
    """Returns the number of values for discrete BoundedArraySpec."""
    if is_discrete(self):
      return (
          np.broadcast_to(self.maximum, shape=self.shape)
          - np.broadcast_to(self.minimum, shape=self.shape)
          + 1
      )

  def __repr__(self):
    template = (
        'BoundedArraySpec(shape={}, dtype={}, name={}, minimum={}, maximum={})'
    )
    return template.format(
        self.shape,
        repr(self.dtype),
        repr(self.name),
        self._minimum,
        self._maximum,
    )

  def __eq__(self, other):
    if not isinstance(other, BoundedArraySpec):
      return False
    return (
        super(BoundedArraySpec, self).__eq__(other)
        and (self.minimum == other.minimum).all()
        and (self.maximum == other.maximum).all()
    )

  def check_array(self, array):
    """Return true if the given array conforms to the spec."""
    return (
        super(BoundedArraySpec, self).check_array(array)
        and np.all(array >= self.minimum)
        and np.all(array <= self.maximum)
    )

  def replace(  # pylint: disable=arguments-renamed
      self, shape=None, dtype=None, minimum=None, maximum=None, name=None
  ):
    shape = self.shape if shape is None else shape
    dtype = self.dtype if dtype is None else dtype
    minimum = self.minimum if minimum is None else minimum
    maximum = self.maximum if maximum is None else maximum
    name = self.name if name is None else name
    return BoundedArraySpec(shape, dtype, minimum, maximum, name)

  def __reduce__(self):
    return (
        BoundedArraySpec,
        (self.shape, self.dtype, self.minimum, self.maximum, self.name),
    )


# from tf_agents.typing.types
Tnest = TypeVar('Tnest')
Trecursive = TypeVar('Trecursive')
# EagerTensor removed to avoid dependency on private TF internals.
# Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor, EagerTensor]
# Array = Union[np.ndarray, int, float, str, bool, EagerTensor]
Tensor = Union[tf.Tensor, tf.SparseTensor, tf.RaggedTensor]
Array = Union[np.ndarray, int, float, str, bool]
TensorSpec = Union[
    tf.TypeSpec,
    tf.TensorSpec,
    tf.RaggedTensorSpec,
    tf.SparseTensorSpec,
    DistributionSpecV2,
]
Nested = Union[Tnest, Iterable[Trecursive], Mapping[Text, Trecursive]]
# Removed NestedDistribution to avoid dependency on TF Probability
# NestedDistribution = Nested[
#     tfp.distributions.Distribution, 'NestedDistribution'
# ]
NestedTensorSpec = Nested[TensorSpec, 'NestedTensorSpec']
NestedArraySpec = Nested[ArraySpec, 'NestedArraySpec']
NestedSpec = Union[NestedTensorSpec, NestedArraySpec]
NestedTensor = Nested[Tensor, 'NestedTensor']
NestedArray = Nested[Array, 'NestedArray']
NestedSpecTensorOrArray = Union[NestedSpec, NestedTensor, NestedArray]
Spec = Union[TensorSpec, ArraySpec]
SpecTensorOrArray = Union[Spec, Tensor, Array]
Bool = Union[bool, bool, Tensor, Array]
NestedTensorOrArray = Union[NestedTensor, NestedArray]
Float = Union[float, np.float16, np.float32, np.float64, Tensor, Array]
TensorOrArray = Union[Tensor, Array]
Shape = Union[TensorOrArray, Sequence[Optional[int]], tf.TensorShape]

# from tf_agents.specs.tensor_spec
TensorSpec = tf.TensorSpec
# BoundedTensorSpec changed to TensorSpec to avoid a dependency on
# private TF internals.
# BoundedTensorSpec = ts.BoundedTensorSpec


# from tf_agents.trajectories.policy_step
# Removed NestedDistribution to avoid dependency on TF Probability
# ActionType = Union[NestedSpecTensorOrArray, NestedDistribution]
ActionType = NestedSpecTensorOrArray


class PolicyStep(
    NamedTuple(
        'PolicyStep',
        [
            ('action', ActionType),
            ('state', NestedSpecTensorOrArray),
            ('info', NestedSpecTensorOrArray),
        ],
    )
):
  """Returned with every call to `policy.action()` and `policy.distribution()`.

  Attributes:
   action: An action tensor or action distribution for `TFPolicy`, or numpy
     array for `PyPolicy`.
   state: During inference, it will hold the state of the policy to be fed back
     into the next  call to policy.action() or policy.distribution(), e.g. an
     RNN state. During the training, it will hold the state that is input to
     policy.action() or policy.distribution() For stateless policies, this will
     be an empty tuple.
   info: Auxiliary information emitted by the policy, e.g. log probabilities of
     the actions. For policies without info this will be an empty tuple.
  """

  __slots__ = ()

  def replace(self, **kwargs) -> 'PolicyStep':
    """Exposes as namedtuple._replace.

    Usage:
    ```
      new_policy_step = policy_step.replace(action=())
    ```

    This returns a new policy step with an empty action.

    Args:
      **kwargs: key/value pairs of fields in the policy step.

    Returns:
      A new `PolicyStep`.
    """
    return self._replace(**kwargs)


# Set default empty tuple for PolicyStep.state and PolicyStep.info.
PolicyStep.__new__.__defaults__ = ((),) * len(PolicyStep._fields)


# from tf_agents.trajectories.time_step
class TimeStep(
    NamedTuple(
        'TimeStep',
        [
            ('step_type', SpecTensorOrArray),
            ('reward', NestedSpecTensorOrArray),
            ('discount', SpecTensorOrArray),
            ('observation', NestedSpecTensorOrArray),
        ],
    )
):
  """Returned with every call to `step` and `reset` on an environment.

  A `TimeStep` contains the data emitted by an environment at each step of
  interaction. A `TimeStep` holds a `step_type`, an `observation` (typically a
  NumPy array or a dict or list of arrays), and an associated `reward` and
  `discount`.

  The first `TimeStep` in a sequence will equal `StepType.FIRST`. The final
  `TimeStep` will equal `StepType.LAST`. All other `TimeStep`s in a sequence
  will equal `StepType.MID.

  Attributes:
    step_type: a `Tensor` or array of `StepType` enum values.
    reward: a `Tensor` or array of reward values.
    discount: A discount value in the range `[0, 1]`.
    observation: A NumPy array, or a nested dict, list or tuple of arrays.
  """

  __slots__ = ()

  def is_first(self) -> Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.FIRST)
    return np.equal(self.step_type, StepType.FIRST)

  def is_mid(self) -> Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.MID)
    return np.equal(self.step_type, StepType.MID)

  def is_last(self) -> Bool:
    if tf.is_tensor(self.step_type):
      return tf.equal(self.step_type, StepType.LAST)
    return np.equal(self.step_type, StepType.LAST)

  def __hash__(self):
    # TODO: Explore performance impact and consider converting
    # dicts in the observation into ordered dicts in __new__ call.
    return hash(tuple(tf.nest.flatten(self)))

  def __repr__(self):
    return (
        'TimeStep(\n'
        + pprint.pformat(dict(self._asdict()), sort_dicts=False)
        + ')'
    )


class StepType(object):
  """Defines the status of a `TimeStep` within a sequence."""

  # Denotes the first `TimeStep` in a sequence.
  FIRST = np.asarray(0, dtype=np.int32)
  # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
  MID = np.asarray(1, dtype=np.int32)
  # Denotes the last `TimeStep` in a sequence.
  LAST = np.asarray(2, dtype=np.int32)

  def __new__(cls, value):
    """Add ability to create StepType constants from a value."""
    if value == cls.FIRST:
      return cls.FIRST
    if value == cls.MID:
      return cls.MID
    if value == cls.LAST:
      return cls.LAST

    raise ValueError('No known conversion for `%r` into a StepType' % value)


def _as_array(a, t=np.float32):
  if t is None:
    t = np.float32
  r = np.asarray(a, dtype=t)
  if np.isnan(np.sum(r)):
    raise ValueError(
        'Received a time_step input that converted to a nan array.'
        ' Did you accidentally set some input value to None?.\n'
        'Got:\n{}'.format(a)
    )
  return r


def _get_np_dtype(spec):
  if not isinstance(spec, (TensorSpec, ArraySpec)):
    return None
  dtype = spec.dtype
  if isinstance(dtype, tf.dtypes.DType):
    dtype = dtype.as_numpy_dtype()
  return np.dtype(dtype)


def transition(
    observation: NestedTensorOrArray,
    reward: NestedTensorOrArray,
    discount: Float = 1.0,
    outer_dims: Optional[Shape] = None,
) -> TimeStep:
  """Returns a `TimeStep` with `step_type` set equal to `StepType.MID`.

  For TF transitions, the batch size is inferred from the shape of `reward`.

  If `discount` is a scalar, and `observation` contains Tensors,
  then `discount` will be broadcasted to match `reward.shape`.

  Args:
    observation: A NumPy array, tensor, or a nested dict, list or tuple of
      arrays or tensors.
    reward: A NumPy array, tensor, or a nested dict, list or tuple of arrays or
      tensors.
    discount: (optional) A scalar, or 1D NumPy array, or tensor.
    outer_dims: (optional) If provided, it will be used to determine the batch
      dimensions. If not, the batch dimensions will be inferred by reward's
      shape. If reward is a vector, but not batched use ().

  Returns:
    A `TimeStep`.

  Raises:
    ValueError: If observations are tensors but reward's statically known rank
      is not `0` or `1`.
  """
  first_observation = tf.nest.flatten(observation)[0]
  if not tf.is_tensor(first_observation):
    if outer_dims is not None:
      step_type = np.tile(StepType.MID, outer_dims)
      discount = _as_array(discount)
      return TimeStep(step_type, reward, discount, observation)
    # Infer the batch size.
    reward = tf.nest.map_structure(
        lambda x: _as_array(x, _get_np_dtype(x)), reward
    )
    first_reward = tf.nest.flatten(reward)[0]
    discount = _as_array(discount)
    if first_reward.shape:
      step_type = np.tile(StepType.MID, first_reward.shape)
    else:
      step_type = StepType.MID
    return TimeStep(step_type, reward, discount, observation)

  # TODO: If reward.shape.rank == 2, and static
  # batch sizes are available for both first_observation and reward,
  # check that these match.
  reward = tf.nest.map_structure(
      # pylint: disable=g-long-lambda
      lambda r: tf.convert_to_tensor(value=r, dtype=r.dtype, name='reward'),
      reward,
  )
  if outer_dims is not None:
    shape = outer_dims
  else:
    first_reward = tf.nest.flatten(reward)[0]
    if first_reward.shape.rank == 0:
      shape = []
    else:
      shape = [
          tf.compat.dimension_value(first_reward.shape[0])
          or tf.shape(input=first_reward)[0]
      ]
  step_type = tf.fill(shape, StepType.MID, name='step_type')
  discount = tf.convert_to_tensor(
      value=discount, dtype=tf.float32, name='discount'
  )
  if discount.shape.rank == 0:
    discount = tf.fill(shape, discount, name='discount_fill')
  return TimeStep(step_type, reward, discount, observation)


def time_step_spec(
    observation_spec: Optional[NestedSpec] = None,
    reward_spec: Optional[NestedSpec] = None,
) -> TimeStep:
  """Returns a `TimeStep` spec given the observation_spec.

  Args:
    observation_spec: A nest of `tf.TypeSpec` or `ArraySpec` objects.
    reward_spec: (Optional) A nest of `tf.TypeSpec` or `ArraySpec` objects.
      Default - a scalar float32 of the same type (Tensor or Array) as
      `observation_spec`.

  Returns:
    A `TimeStep` with the same types (`TypeSpec` or `ArraySpec`) as
    the first entry in `observation_spec`.

  Raises:
    TypeError: If observation and reward specs aren't both either tensor type
      specs or array type specs.
  """
  if observation_spec is None:
    # Changed to pass pytype checks not accepting empty tuples.
    #
    # return TimeStep(step_type=(), reward=(), discount=(), observation=())
    return TimeStep(
        step_type=ArraySpec([], np.int32, name='step_type'),
        reward=(),
        discount=BoundedArraySpec(
            [], np.float32, minimum=0.0, maximum=1.0, name='discount'
        ),
        observation=(),
    )

  first_observation_spec = tf.nest.flatten(observation_spec)[0]
  if reward_spec is not None:
    first_reward_spec = tf.nest.flatten(reward_spec)[0]
    if isinstance(first_reward_spec, tf.TypeSpec) != isinstance(
        first_observation_spec, tf.TypeSpec
    ):
      raise TypeError(
          'Expected observation and reward specs to both be either tensor or '
          'array specs, but saw spec values {} vs. {}'.format(
              first_observation_spec, first_reward_spec
          )
      )
  if isinstance(first_observation_spec, tf.TypeSpec):
    return TimeStep(
        step_type=TensorSpec([], tf.int32, name='step_type'),
        reward=reward_spec or tf.TensorSpec([], tf.float32, name='reward'),
        # BoundedTensorSpec changed to TensorSpec to avoid a dependency on
        # private TF internals.
        #
        # discount=BoundedTensorSpec(
        #     [], tf.float32, minimum=0.0, maximum=1.0, name='discount'
        # ),
        discount=TensorSpec([], tf.float32, name='discount'),
        observation=observation_spec,
    )
  return TimeStep(
      step_type=ArraySpec([], np.int32, name='step_type'),
      reward=reward_spec or ArraySpec([], np.float32, name='reward'),
      discount=BoundedArraySpec(
          [], np.float32, minimum=0.0, maximum=1.0, name='discount'
      ),
      observation=observation_spec,
  )
