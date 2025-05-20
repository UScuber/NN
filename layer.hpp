#pragma once
#include <cmath>
#include <algorithm>
#include "matrix.hpp"

// ユニットと1つ目のレイヤーからの重みを持つ
struct Layer {
  // Inputの場合、pre_size=0
  Layer(const int size, const int pre_size) : size(size), pre_size(pre_size){}

  virtual ~Layer(){}

  // x(g): 1つ前の出力
  // w: 1つ前のlayer -> this
  std::vector<double> forward(const std::vector<double> &x, const matrix<double> &w){
    assert(w.height() == (int)x.size());

    h = x * w;
    assert((int)h.size() == size);
    pre_g = x; // backward用
    std::vector<double> g(size);
    for(int i = 0; i < size; i++){
      g[i] = f(h[i]);
    }
    return g;
  }

  // next_e: 1つ先のLayerの誤差
  // rho: 学習率
  // return: e
  std::vector<double> backward(const std::vector<double> &next_e, matrix<double> &w, const double rho){
    assert((int)next_e.size() == size);
    assert(w.height() == pre_size && w.width() == size);

    std::vector<double> eps(size);
    for(int j = 0; j < size; j++){
      eps[j] = next_e[j] * df(h[j]);
    }

    std::vector<double> e(pre_size);
    for(int i = 0; i < pre_size; i++){
      for(int j = 0; j < size; j++){
        e[i] += next_e[j] * w[i][j];
      }
    }
    
    // update
    for(int i = 0; i < pre_size; i++){
      for(int j = 0; j < size; j++){
        w[i][j] -= rho * pre_g[i] * eps[j];
      }
    }
    return e;
  }

  int size, pre_size;
private:
  virtual double f(const double x) = 0;
  virtual double df(const double x) = 0;
protected:
  std::vector<double> pre_g, h;
};

struct Input : Layer {
  using Layer::Layer;
private:
  double f(const double x){
    return x;
  }
  
  double df(const double){
    return 1;
  }
};

struct Output : Layer {
  using Layer::Layer;
private:
  double f(const double x){
    return x;
  }

  double df(const double){
    return 1;
  }
};

struct ReLU : Layer {
  using Layer::Layer;
private:
  double f(const double x){
    return std::max(0.0, x);
  }

  double df(const double x){
    return x >= 0.0;
  }
};

struct Sigmoid : Layer {
  using Layer::Layer;
private:
  double f(const double x){
    return 1.0 / (1.0 + exp(-x));
  }

  double df(const double x){
    return f(x) * (1.0 - f(x));
  }
};
