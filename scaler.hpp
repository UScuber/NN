#pragma once
#include <cmath>
#include "matrix.hpp"

struct StandardScaler {
  StandardScaler(){}

  void fit(const matrix<double> &data){
    if(mu.empty()){
      mu.assign(data.width(), 0.0);
      sd.assign(data.width(), 0.0);
    }
    assert((int)mu.size() == data.width() && (int)sd.size() == data.width());

    for(int j = 0; j < data.width(); j++){
      double avg = 0;
      for(int i = 0; i < data.height(); i++){
        avg += data[i][j];
      }
      avg /= data.height();
      double var = 0;
      for(int i = 0; i < data.height(); i++){
        var += (data[i][j] - avg) * (data[i][j] - avg);
      }
      mu[j] = avg;
      sd[j] = sqrt(var / data.height());
    }
  }

  void transform(matrix<double> &data){
    assert((int)mu.size() == data.width());
    for(int i = 0; i < data.height(); i++){
      for(int j = 0; j < data.width(); j++){
        data[i][j] = (data[i][j] - mu[j]) / sd[j];
      }
    }
  }

private:
  std::vector<double> mu, sd;
};


struct MinMaxScaler {
  MinMaxScaler(){}

  void fit(const matrix<double> &data){
    if(mi.empty()){
      mi.assign(data.width(), 1e9);
      mx.assign(data.width(), -1e9);
    }
    assert((int)mi.size() == data.width() && (int)mx.size() == data.width());

    for(int i = 0; i < data.height(); i++){
      for(int j = 0; j < data.width(); j++){
        mi[j] = std::min(mi[j], data[i][j]);
        mx[j] = std::max(mx[j], data[i][j]);
      }
    }
  }

  void transform(matrix<double> &data){
    assert((int)mi.size() == data.width());
    for(int i = 0; i < data.height(); i++){
      for(int j = 0; j < data.width(); j++){
        data[i][j] = (data[i][j] - mi[j]) / (mx[j] - mi[j]);
      }
    }
  }

private:
  std::vector<double> mi, mx;
};
