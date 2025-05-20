#pragma once

#include <fstream>
#include <sstream>
#include "matrix.hpp"

void load_dataset(matrix<double> &X_data, std::vector<int> &y_data, const std::string &filename){
  std::ifstream file(filename);
  std::vector<std::vector<double>> mat;
  std::string line;

  if(!file){
    std::cerr << "cannot open file: " << filename << "\n";
    exit(1);
  }

  int width = -1;

  while(std::getline(file, line)){
    std::istringstream iss(line);
    std::vector<double> row;
    double value;

    while(iss >> value){
      row.push_back(value);
    }

    if(row.empty()) continue;

    assert(width == -1 || width == (int)row.size());
    if(width == -1) width = row.size();

    mat.push_back(row);
  }

  assert(!mat.empty() && width >= 2);

  X_data = matrix<double>(mat.size(), (int)mat[0].size()-1);
  y_data = std::vector<int>(mat.size());
  for(int i = 0; i < (int)mat.size(); i++){
    X_data[i] = std::vector<double>(mat[i].begin(), mat[i].end()-1);
    y_data[i] = mat[i].back();
  }
}
