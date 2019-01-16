#ifndef __DATA_LOADER__
#define DATA_LOADER

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>

class DataLoader
{
public:
    // BCHW
    DataLoader(int batchSize, std::string file_list, int width, int height, int channel) {
        _cur_id = 0;
        _file_list = file_list;
        _bsize = batchSize;
        _width = width;
        _height = height;
        _channel = channel;
        _batch = new float[ _bsize*_width*_height*_channel];
        std::ifstream infile(file_list);
        std::string tmp;
        while(infile >> tmp) {
            _fnames.push_back(tmp);
        }
    }
    void reset(){
        _cur_id = 0;
    }
    float *getBatch() { return  _batch; }
    bool next(){
        std::cout<< "calling next(). Cur_id: "<< _cur_id<<std::endl;
        if(_cur_id + _bsize >= _fnames.size()){
            return false;
        }
        for (unsigned int i = 0; i < _bsize; ++i) {
            std::string fname = _fnames[i];
            cv::Mat img = cv::imread(fname, -1);
            int ww = img.cols;
            int hh = img.rows;
            int cc = img.channels();
            float pixelMean[3]{ 102.9801f, 115.9465f, 122.7717f };

            for(int c=0; c<cc; c++){
                for(int h=0; h<hh; h++){
                    for(int w=0; w<ww; w++){
                        int pix_id = h*w*c + w*c + c;
                        int row_id = i*h*w*c + c*h*w + h*w + w;
                        _batch[row_id] = float(img.ptr()[pix_id] - pixelMean[c]);
                    }
                }
            }
            img.relase();
        }
        _cur_id += _bsize;
        return true;
    }

    ~DataLoader(){
        delete[] _batch;
        delete[] _im_info;
    }

private:
    unsigned int _cur_id;
    std::string _file_list;
    std::vector<std::string> _fnames;
    float* _batch;
    unsigned int _bsize;
    int _width;
    int _height;
    int _channel;

};

#endif
