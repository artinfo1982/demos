CPP=g++

test:
  $(CPP) src/Main.cpp src/PlrIntoSrc.h src/PlrIntoSrc.cpp -o bin/test -O2 `pkg-config --libs opencv`
  
clean:
  rm bin/test
  rm imageset/train/*.jpg
  rm xml/*.xml
