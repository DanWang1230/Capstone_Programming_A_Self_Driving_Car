# This Python file uses the following encoding: utf-8
"""autogenerated by genpy from dbw_mkz_msgs/Wiper.msg. Do not edit."""
import sys
python3 = True if sys.hexversion > 0x03000000 else False
import genpy
import struct


class Wiper(genpy.Message):
  _md5sum = "7fccb48d5d1df108afaa89f8fc14ce1c"
  _type = "dbw_mkz_msgs/Wiper"
  _has_header = False #flag to mark the presence of a Header object
  _full_text = """uint8 status

uint8 OFF=0
uint8 AUTO_OFF=1
uint8 OFF_MOVING=2
uint8 MANUAL_OFF=3
uint8 MANUAL_ON=4
uint8 MANUAL_LOW=5
uint8 MANUAL_HIGH=6
uint8 MIST_FLICK=7
uint8 WASH=8
uint8 AUTO_LOW=9
uint8 AUTO_HIGH=10
uint8 COURTESYWIPE=11
uint8 AUTO_ADJUST=12
uint8 RESERVED=13
uint8 STALLED=14
uint8 NO_DATA=15
"""
  # Pseudo-constants
  OFF = 0
  AUTO_OFF = 1
  OFF_MOVING = 2
  MANUAL_OFF = 3
  MANUAL_ON = 4
  MANUAL_LOW = 5
  MANUAL_HIGH = 6
  MIST_FLICK = 7
  WASH = 8
  AUTO_LOW = 9
  AUTO_HIGH = 10
  COURTESYWIPE = 11
  AUTO_ADJUST = 12
  RESERVED = 13
  STALLED = 14
  NO_DATA = 15

  __slots__ = ['status']
  _slot_types = ['uint8']

  def __init__(self, *args, **kwds):
    """
    Constructor. Any message fields that are implicitly/explicitly
    set to None will be assigned a default value. The recommend
    use is keyword arguments as this is more robust to future message
    changes.  You cannot mix in-order arguments and keyword arguments.

    The available fields are:
       status

    :param args: complete set of field values, in .msg order
    :param kwds: use keyword arguments corresponding to message field names
    to set specific fields.
    """
    if args or kwds:
      super(Wiper, self).__init__(*args, **kwds)
      #message fields cannot be None, assign default values for those that are
      if self.status is None:
        self.status = 0
    else:
      self.status = 0

  def _get_types(self):
    """
    internal API method
    """
    return self._slot_types

  def serialize(self, buff):
    """
    serialize message into buffer
    :param buff: buffer, ``StringIO``
    """
    try:
      buff.write(_get_struct_B().pack(self.status))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize(self, str):
    """
    unpack serialized message in str into this message instance
    :param str: byte array of serialized message, ``str``
    """
    try:
      end = 0
      start = end
      end += 1
      (self.status,) = _get_struct_B().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill


  def serialize_numpy(self, buff, numpy):
    """
    serialize message with numpy array types into buffer
    :param buff: buffer, ``StringIO``
    :param numpy: numpy python module
    """
    try:
      buff.write(_get_struct_B().pack(self.status))
    except struct.error as se: self._check_types(struct.error("%s: '%s' when writing '%s'" % (type(se), str(se), str(locals().get('_x', self)))))
    except TypeError as te: self._check_types(ValueError("%s: '%s' when writing '%s'" % (type(te), str(te), str(locals().get('_x', self)))))

  def deserialize_numpy(self, str, numpy):
    """
    unpack serialized message in str into this message instance using numpy for array types
    :param str: byte array of serialized message, ``str``
    :param numpy: numpy python module
    """
    try:
      end = 0
      start = end
      end += 1
      (self.status,) = _get_struct_B().unpack(str[start:end])
      return self
    except struct.error as e:
      raise genpy.DeserializationError(e) #most likely buffer underfill

_struct_I = genpy.struct_I
def _get_struct_I():
    global _struct_I
    return _struct_I
_struct_B = None
def _get_struct_B():
    global _struct_B
    if _struct_B is None:
        _struct_B = struct.Struct("<B")
    return _struct_B
