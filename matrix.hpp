#pragma once
#include <iostream>
#include <vector>
#include <cassert>

template <typename T>
struct matrix {
  matrix(const int n=0) : n(n), m(n), a(n, std::vector<T>(n)){}

  matrix(const int n, const int m) : n(n), m(m), a(n, std::vector<T>(m)){}

  matrix(const std::vector<std::vector<T>> &d) : a(d), n(d.size()), m(d[0].size()){}

  std::vector<T> &operator[](const int i){ return a[i]; }

  const std::vector<T> &operator[](const int i) const{ return a[i]; }

  matrix &operator*=(const matrix &b){
    assert(m == b.n);
    std::vector<std::vector<T>> c(n, std::vector<T>(b.m));
    for(int i = 0; i < n; i++) for(int j = 0; j < m; j++)
    for(int k = 0; k < b.m; k++){
      c[i][k] += a[i][j] * b.a[j][k];
    }
    a = c;
    return *this;
  }

  matrix &operator+=(const matrix &b){
    assert(n == b.n && m == b.m);
    for(int i = 0; i < n; i++) for(int j = 0; j < m; j++)
      a[i][j] += b.a[i][j];
    return *this;
  }

  matrix &operator-=(const matrix &b){
    assert(n == b.n && m == b.m);
    for(int i = 0; i < n; i++) for(int j = 0; j < m; j++)
      a[i][j] -= b.a[i][j];
    return *this;
  }

  matrix operator*(const matrix &b) const{ return matrix(*this) *= b; }

  std::vector<T> operator*(const std::vector<T> &b){
    assert(m == (int)b.size());
    std::vector<T> c(n);
    for(int i = 0; i < n; i++){
      for(int j = 0; j < m; j++){
        c[i] += a[i][j] * b[j];
      }
    }
    return c;
  }

  matrix operator+(const matrix &b) const{ return matrix(*this) += b; }

  matrix operator-(const matrix &b) const{ return matrix(*this) -= b; }

  matrix pow(long long t) const{
    assert(n == m);
    matrix<T> b(n), c = *this;
    for(int i = 0; i < n; i++) b[i][i] = 1;
    while(t > 0){
      if(t & 1) b *= c;
      c *= c;
      t >>= 1;
    }
    return b;
  }

  T det() const{
    assert(n == m);
    matrix b = *this;
    T res(1);
    bool flip = false;
    for(int i = 0; i < n; i++){
      for(int j = i + 1; j < n; j++){
        while(b[j][i] > 0){
          swap(b[i], b[j]);
          flip ^= 1;
          const T d = b[j][i] / b[i][i];
          for(int k = i; k < n; k++){
            b[j][k] -= b[i][k] * d;
          }
        }
      }
      if(b[i][i] == 0) return 0;
      res *= b[i][i];
    }
    if(flip) res = -res;
    return res;
  }

  matrix inv(){
    assert(n == m);
    matrix b(n), c = *this;
    for(int i = 0; i < n; i++) b[i][i] = 1;
    int r = 0;
    for(int i = 0; i < n && r < n; i++){
      if(c[r][i] == 0){
        T max_val = 0; int mx_pos;
        for(int j = r+1; j < n; j++){
          if(max_val < c[j][i]) max_val = c[j][i], mx_pos = j;
        }
        if(max_val == 0) return false;
        swap(c[r], c[mx_pos]); swap(b[r], b[mx_pos]);
      }     
      T d = T(1) / c[r][i];
      for(int j = 0; j < n; j++) c[r][j] *= d, b[r][j] *= d;
      for(int j = 0; j < n; j++){
        T v = c[j][i];
        if(j == r || c[j][i] == 0) continue;
        for(int k = 0; k < n; k++){
          c[j][k] -= c[r][k] * v;
          b[j][k] -= b[r][k] * v;
        }
      }
      r++;
    }
    return b;
  }

  matrix transpose(){
    matrix t(m, n);
    for(int i = 0; i < n; i++){
      for(int j = 0; j < m; j++) t[j][i] = a[i][j];
    }
    return t;
  }

  int height() const{ return n; }
  int width()  const{ return m; }

  void debug(){
    for(int i = 0; i < n; i++){
      for(int j = 0; j < m; j++) std::cerr << a[i][j] << " ";
      std::cerr << "\n";
    }
  }

  int n,m;
private:
  std::vector<std::vector<T>> a;
};

template <typename T>
std::vector<T> operator*(const std::vector<T> &a, const matrix<T> &b){
  assert((int)a.size() == b.height());
  std::vector<T> c(b.width());
  for(int i = 0; i < b.height(); i++){
    for(int j = 0; j < b.width(); j++){
      c[j] += a[i] * b[i][j];
    }
  }
  return c;
}

double dot(const std::vector<double> &a, const std::vector<double> &b){
  assert(a.size() == b.size());
  double res = 0;
  for(int i = 0; i < (int)a.size(); i++) res += a[i] * b[i];
  return res;
}
