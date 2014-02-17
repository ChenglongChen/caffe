#ifndef CAFFE_NET_INFO_HPP_
#define CAFFE_NET_INFO_HPP_

#include <vector>
#include <iomanip>

using std::setw;
using std::scientific;
using std::left;

namespace caffe {

template <typename Dtype>
class Info {
 public:
  explicit Info() {};
  virtual void Print(Net<Dtype>& net) = 0;
  virtual ~Info() {};
};

template <typename Dtype>
class WeightRelatedInfo : public Info<Dtype> {
 public:
  explicit WeightRelatedInfo() {};
  virtual void Print(Net<Dtype>& net) {
    int max_len = 0;
    for (int l = 0; l < net.layers().size(); ++l) {
      Layer<Dtype>& layer = *net.layers()[l].get();
      if (layer.blobs().size() > 0 && layer.layer_param().name().length() > max_len)
	max_len = layer.layer_param().name().length();
    }
    for (int l = 0; l < net.layers().size(); ++l) {
      Layer<Dtype>& layer = *net.layers()[l].get();
      for (int b = 0; b < layer.blobs().size(); ++b) {
	Blob<Dtype>& blob = *layer.blobs()[b].get();
	const Dtype* blob_cpu_data = blob.cpu_data();
	const Dtype* blob_cpu_diff = blob.cpu_diff();
	Dtype data_sum = 0;
	Dtype diff_sum = 0;
	for (int i = 0; i < blob.count(); ++i) {
	  data_sum += (blob_cpu_data[i] > Dtype(0.)) ? blob_cpu_data[i] : - blob_cpu_data[i];
	  diff_sum += (blob_cpu_diff[i] > Dtype(0.)) ? blob_cpu_diff[i] : - blob_cpu_diff[i];
	}
	data_sum /= blob.count();
	diff_sum /= blob.count();
	LOG(INFO) << std::left << std::setw(max_len + 1) << std::setfill(' ')
		  << layer.layer_param().name()
		  << " blob" << b << ": " << std::scientific 
		  << data_sum << " [" << diff_sum << "]";
      }
    }
  };
  virtual ~WeightRelatedInfo() {};
};

template <typename Dtype>
class BlobRelatedInfo : public Info<Dtype> {
 public:
  explicit BlobRelatedInfo() {};
  virtual void Print(Net<Dtype>& net) {
    int max_len = 0;
    for (int l = 0; l < net.blob_names().size(); ++l) {
      if (net.blob_names()[l].length() > max_len)
	max_len = net.blob_names()[l].length();
    }
    for (int l = 0; l < net.blobs().size(); ++l) {
      Blob<Dtype>& blob = *net.blobs()[l].get();
      std::string blob_name = net.blob_names()[l];
      const Dtype* blob_cpu_data = blob.cpu_data();
      const Dtype* blob_cpu_diff = blob.cpu_diff();
      Dtype data_max = 0;
      Dtype data_min = 0;
      Dtype diff_max = 0;
      Dtype diff_min = 0;
      for (int i = 0; i < blob.count(); ++i) {
	data_max = (blob_cpu_data[i] > data_max) ? blob_cpu_data[i] : data_max;
	data_min = (blob_cpu_data[i] < data_min) ? blob_cpu_data[i] : data_min;
	diff_max = (blob_cpu_diff[i] > diff_max) ? blob_cpu_diff[i] : diff_max;
	diff_min = (blob_cpu_diff[i] < diff_min) ? blob_cpu_diff[i] : diff_min;
      }
      LOG(INFO) << std::left << std::setw(max_len + 1) << std::setfill(' ')
		<< net.blob_names()[l] << std::scientific
		<< " data: (" << data_max << ", " << data_min << ") "
		<< "diff: (" << diff_max << ", " << diff_min << ")";
    }
  }
  virtual ~BlobRelatedInfo() {}
};

template <typename Dtype>
Info<Dtype>* GetInfo(const std::string& type) {
  if (type == "weight") {
    return new WeightRelatedInfo<Dtype>();
  } else if (type == "blob") {
    return new BlobRelatedInfo<Dtype>();
  } else {
    CHECK(false) << "Unknown info type: " << type;
  }
  return (Info<Dtype>*)(NULL);
}

}  // namspace caffe

#endif  // CAFFE_NET_INFO_HPP_
