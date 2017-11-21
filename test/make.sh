g++ init_cnn.cpp -o init_cnn.so -fPIC -shared -pthread -O3 -march=native
g++ init_know.cpp -o init_know.so -fPIC -shared -pthread -O3 -march=native
g++ test.cpp -o test -O3 -march=native
