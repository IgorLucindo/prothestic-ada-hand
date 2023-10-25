// Generated by gencpp from file ada_visual_control/FloatArray.msg
// DO NOT EDIT!


#ifndef ADA_VISUAL_CONTROL_MESSAGE_FLOATARRAY_H
#define ADA_VISUAL_CONTROL_MESSAGE_FLOATARRAY_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace ada_visual_control
{
template <class ContainerAllocator>
struct FloatArray_
{
  typedef FloatArray_<ContainerAllocator> Type;

  FloatArray_()
    : data()  {
    }
  FloatArray_(const ContainerAllocator& _alloc)
    : data(_alloc)  {
  (void)_alloc;
    }



   typedef std::vector<float, typename std::allocator_traits<ContainerAllocator>::template rebind_alloc<float>> _data_type;
  _data_type data;





  typedef boost::shared_ptr< ::ada_visual_control::FloatArray_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::ada_visual_control::FloatArray_<ContainerAllocator> const> ConstPtr;

}; // struct FloatArray_

typedef ::ada_visual_control::FloatArray_<std::allocator<void> > FloatArray;

typedef boost::shared_ptr< ::ada_visual_control::FloatArray > FloatArrayPtr;
typedef boost::shared_ptr< ::ada_visual_control::FloatArray const> FloatArrayConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::ada_visual_control::FloatArray_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::ada_visual_control::FloatArray_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::ada_visual_control::FloatArray_<ContainerAllocator1> & lhs, const ::ada_visual_control::FloatArray_<ContainerAllocator2> & rhs)
{
  return lhs.data == rhs.data;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::ada_visual_control::FloatArray_<ContainerAllocator1> & lhs, const ::ada_visual_control::FloatArray_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace ada_visual_control

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::ada_visual_control::FloatArray_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::ada_visual_control::FloatArray_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ada_visual_control::FloatArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::ada_visual_control::FloatArray_<ContainerAllocator> const>
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ada_visual_control::FloatArray_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::ada_visual_control::FloatArray_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::ada_visual_control::FloatArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "420cd38b6b071cd49f2970c3e2cee511";
  }

  static const char* value(const ::ada_visual_control::FloatArray_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0x420cd38b6b071cd4ULL;
  static const uint64_t static_value2 = 0x9f2970c3e2cee511ULL;
};

template<class ContainerAllocator>
struct DataType< ::ada_visual_control::FloatArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "ada_visual_control/FloatArray";
  }

  static const char* value(const ::ada_visual_control::FloatArray_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::ada_visual_control::FloatArray_<ContainerAllocator> >
{
  static const char* value()
  {
    return "float32[] data\n"
;
  }

  static const char* value(const ::ada_visual_control::FloatArray_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::ada_visual_control::FloatArray_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.data);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct FloatArray_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::ada_visual_control::FloatArray_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::ada_visual_control::FloatArray_<ContainerAllocator>& v)
  {
    s << indent << "data[]" << std::endl;
    for (size_t i = 0; i < v.data.size(); ++i)
    {
      s << indent << "  data[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.data[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // ADA_VISUAL_CONTROL_MESSAGE_FLOATARRAY_H
