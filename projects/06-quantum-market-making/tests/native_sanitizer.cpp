#include "../market_making/native/engine.cpp"
#include <cassert>
#include <random>
int main() {
    Engine e(.01,-.2,2.,.0001,12);
    std::mt19937 rng(1729);
    for(int64_t i=0;i<5000;++i) {
        e.depth(1,9999,rng()%20); e.depth(-1,10001,rng()%20);
        e.quote(i,10000,1,.25,8,2,1,8,1000,1000,i);
        for(int side:{1,-1}) {
            e.add(i,100000+i*2+(side==1),side,1,1+rng()%8,false,true);
            assert(std::abs(e.inventory)<=12);
        }
        if(i%7==0) e.cancel_all(false);
    }
    e.cancel_all(true);
    assert(e.open(1)==0 && e.open(-1)==0);
}
