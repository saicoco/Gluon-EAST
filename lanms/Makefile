# CXXFLAGS = -I include  -std=c++11 -O3 $(shell python-config --cflags)
# LDFLAGS = $(shell python-config --ldflags)

CXXFLAGS = -I include  -std=c++11 -O3  -I/Users/gengjiajia/miniconda2/include/python2.7

LDFLAGS = $(shell python-config --ldflags) -lpython2.7 -L/Users/gengjiajia/miniconda2/lib

DEPS = lanms.h $(shell find include -xtype f)
CXX_SOURCES = adaptor.cpp include/clipper/clipper.cpp

LIB_SO = adaptor.so

$(LIB_SO): $(CXX_SOURCES) $(DEPS)
	$(CXX) -o $@ $(CXXFLAGS) $(LDFLAGS) $(CXX_SOURCES) --shared -fPIC

clean:
	rm -rf $(LIB_SO)
