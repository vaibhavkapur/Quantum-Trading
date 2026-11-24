#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <list>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

struct Fill { int64_t timestamp_ns, order_id, side, price, quantity, maker; double fee; };
struct Stats {
    int64_t inventory, open_buy, open_sell, cancels, fills, halted, policy_version;
    double cash, maker_fees, taker_fees, peak_equity;
};
struct Order { int64_t id, side, price, quantity; bool own; };
using Levels = std::map<int64_t, std::list<Order>>;
static thread_local std::string last_error;

class Engine {
public:
    Levels bids, asks;
    std::vector<Fill> fills;
    int64_t inventory=0, cancels=0, version=0, next_id=-1, limit;
    double cash=0, maker_fees=0, taker_fees=0, peak=0;
    double tick, maker_bps, taker_bps, cancel_cost;
    bool halted=false;
    Engine(double t, double m, double k, double c, int64_t l)
        :limit(l),tick(t),maker_bps(m),taker_bps(k),cancel_cost(c) {
        if (!(t>0) || !std::isfinite(t+m+k+c) || c<0 || l<=0 || l>1000000)
            throw std::invalid_argument("invalid engine configuration");
    }
    Levels& book(int64_t side) { return side==1 ? bids : asks; }
    static void valid(int64_t side, int64_t price, int64_t qty) {
        if ((side!=1 && side!=-1) || price<=0 || price>1000000000 || qty<=0 || qty>1000000)
            throw std::invalid_argument("invalid side, price or quantity");
    }
    int64_t open(int side) const {
        int64_t n=0; const auto& levels=side==1 ? bids : asks;
        for (const auto& [p, orders]: levels) for (const auto& o:orders) if(o.own) n+=o.quantity;
        return n;
    }
    bool exists(int64_t id) const {
        for (auto* levels:{&bids,&asks}) for(const auto& [p,orders]:*levels)
            for(const auto& o:orders) if(o.id==id) return true;
        return false;
    }
    bool cancel(int64_t id) {
        for (auto* levels:{&bids,&asks}) for(auto level=levels->begin();level!=levels->end();++level)
            for(auto o=level->second.begin();o!=level->second.end();++o) if(o->id==id) {
                if(o->own) { ++cancels; cash-=cancel_cost; }
                level->second.erase(o); if(level->second.empty()) levels->erase(level); return true;
            }
        return false;
    }
    void cancel_all(bool emergency) {
        std::vector<int64_t> ids;
        for(auto* levels:{&bids,&asks}) for(const auto& [p,orders]:*levels)
            for(const auto& o:orders) if(o.own) ids.push_back(o.id);
        for(auto id:ids) cancel(id);
        if(emergency) halted=true;
    }
    void record(int64_t ts, const Order& own, int64_t price, int64_t qty, bool maker) {
        const double notional=price*tick*qty;
        const double fee=notional*(maker ? maker_bps:taker_bps)*1e-4;
        inventory+=own.side*qty; cash-=own.side*notional+fee;
        (maker ? maker_fees:taker_fees)+=fee;
        fills.push_back({ts,own.id,own.side,price,qty,maker?1:0,fee});
        if(std::abs(inventory)>limit) throw std::logic_error("inventory limit invariant violated");
    }
    int64_t add(int64_t ts,int64_t id,int64_t side,int64_t price,int64_t qty,bool own,bool market=false,bool liquidation=false) {
        valid(side,price,qty);
        if(exists(id)) throw std::invalid_argument("duplicate active order id");
        if(own && !liquidation) {
            if(halted) throw std::runtime_error("risk halt is latched");
            if((side==1 && inventory+open(1)+qty>limit) || (side==-1 && inventory-open(-1)-qty < -limit))
                throw std::invalid_argument("order exceeds worst-case inventory limit");
        }
        auto& opposite=book(-side);
        // Reject self-trades before mutating either side.
        if(own) for(const auto& [p, orders]:opposite) {
            if(!market && (side==1 ? p>price:p<price)) continue;
            for(const auto& o:orders) if(o.own) throw std::invalid_argument("self-trade rejected");
        }
        Order incoming{id,side,price,qty,own};
        while(incoming.quantity>0 && !opposite.empty()) {
            auto level=side==1 ? opposite.begin():std::prev(opposite.end());
            const int64_t p=level->first;
            if(!market && (side==1 ? p>price:p<price)) break;
            auto& resting=level->second.front();
            const int64_t n=std::min(incoming.quantity,resting.quantity);
            if(resting.own) record(ts,resting,p,n,true);
            if(incoming.own) record(ts,incoming,p,n,false);
            incoming.quantity-=n; resting.quantity-=n;
            if(resting.quantity==0) level->second.pop_front();
            if(level->second.empty()) opposite.erase(level);
        }
        if(incoming.quantity>0 && !market) book(side)[price].push_back(incoming);
        return qty-incoming.quantity;
    }
    int64_t queue(int64_t id) const {
        for(auto* levels:{&bids,&asks}) for(const auto& [p,orders]:*levels) {
            int64_t ahead=0; for(const auto& o:orders) { if(o.id==id) return ahead; ahead+=o.quantity; }
        }
        return -1;
    }
    void depth(int side,int64_t price,int64_t qty) {
        if((side!=1 && side!=-1) || price<=0 || price>1000000000 || qty<0 || qty>1000000)
            throw std::invalid_argument("invalid depth");
        auto& levels=book(side); auto found=levels.find(price);
        int64_t current=0;
        if(found!=levels.end()) for(const auto& o:found->second) if(!o.own) current+=o.quantity;
        if(current>qty) {
            // L2 has no order IDs: assume newest external quantity cancels first.
            auto& orders=found->second; int64_t remove=current-qty;
            for(auto it=orders.end();it!=orders.begin() && remove>0;) {
                --it; if(it->own) continue;
                int64_t n=std::min(remove,it->quantity); it->quantity-=n; remove-=n;
                if(it->quantity==0) it=orders.erase(it);
            }
            if(orders.empty()) levels.erase(found);
        } else if(current<qty) {
            // A snapshot is not a trade. Crossed stale own quotes are handled by replay before reconciliation.
            levels[price].push_back({next_id--,side,price,qty-current,false});
        }
    }
    void quote(int64_t ts,double mid,double spread,double skew,int64_t band,int64_t size,
               double sigma,double max_sigma,double max_loss,double max_dd,int64_t new_version) {
        if(!std::isfinite(mid+spread+skew+sigma+max_sigma+max_loss+max_dd) || mid<2 || mid>999999000 ||
           spread<1 || spread>1000000 || skew<0 || skew>1000000 || band<=0 || band>limit || size<=0 || size>limit || sigma<0 ||
           max_sigma<=0 || max_loss<=0 || max_dd<=0 || new_version<0)
            throw std::invalid_argument("invalid quote policy or market state");
        const double equity=cash+inventory*mid*tick; peak=std::max(peak,equity);
        if(equity<=-max_loss || peak-equity>=max_dd || sigma>max_sigma) cancel_all(true);
        if(halted) return;
        if(new_version<version) throw std::invalid_argument("stale policy version");
        version=new_version;
        int64_t bid=std::clamp<int64_t>(static_cast<int64_t>(std::floor(mid-spread-inventory*skew)),1,999999999);
        int64_t ask=std::clamp<int64_t>(static_cast<int64_t>(std::ceil(mid+spread-inventory*skew)),bid+1,1000000000);
        // Post-only clamping against the full visible book.
        if(!asks.empty()) bid=std::min(bid,asks.begin()->first-1);
        if(!bids.empty()) ask=std::max(ask,bids.rbegin()->first+1);
        const int64_t buy=std::min(size,std::max<int64_t>(0,band-inventory));
        const int64_t sell=std::min(size,std::max<int64_t>(0,band+inventory));
        for(int side:{1,-1}) {
            int64_t p=side==1?bid:ask, q=side==1?buy:sell;
            std::vector<int64_t> ids; bool keep=false;
            for(const auto& [level,orders]:book(side)) for(const auto& o:orders) if(o.own) {
                if(!keep && o.price==p && o.quantity==q && q>0) keep=true;
                else ids.push_back(o.id);
            }
            for(auto id:ids) cancel(id);
            if(!keep && q>0 && p>0) add(ts,next_id--,side,p,q,true);
        }
    }
    Stats stats() const {
        return {inventory,open(1),open(-1),cancels,static_cast<int64_t>(fills.size()),halted?1:0,version,
                cash,maker_fees,taker_fees,peak};
    }
};

#define GUARD_BEGIN try { last_error.clear();
#define GUARD_END } catch(const std::exception& e) {last_error=e.what();return -1;} catch(...) {last_error="native failure";return -1;}
extern "C" {
const char* mm_error() {return last_error.c_str();}
void* mm_create(double tick,double maker,double taker,double cancel,int64_t limit) {
    try {return new Engine(tick,maker,taker,cancel,limit);} catch(const std::exception& e) {last_error=e.what();return nullptr;}
}
void mm_destroy(void* p) {delete static_cast<Engine*>(p);}
int64_t mm_add(void* p,int64_t ts,int64_t id,int64_t side,int64_t price,int64_t qty,int own,int market) {
    GUARD_BEGIN return static_cast<Engine*>(p)->add(ts,id,side,price,qty,own,market); GUARD_END
}
int64_t mm_cancel(void* p,int64_t id) {GUARD_BEGIN return static_cast<Engine*>(p)->cancel(id); GUARD_END}
int64_t mm_cancel_all(void* p,int emergency) {GUARD_BEGIN static_cast<Engine*>(p)->cancel_all(emergency);return 0; GUARD_END}
int64_t mm_depth(void* p,int side,int64_t price,int64_t qty) {GUARD_BEGIN static_cast<Engine*>(p)->depth(side,price,qty);return 0; GUARD_END}
int64_t mm_queue(void* p,int64_t id) {return static_cast<Engine*>(p)->queue(id);}
int64_t mm_stats(void* p,Stats* out) {GUARD_BEGIN *out=static_cast<Engine*>(p)->stats();return 0; GUARD_END}
int64_t mm_fill(void* p,int64_t index,Fill* out) {GUARD_BEGIN *out=static_cast<Engine*>(p)->fills.at(index);return 0; GUARD_END}
int64_t mm_quote(void* p,int64_t ts,double mid,double spread,double skew,int64_t band,int64_t size,
                 double sigma,double max_sigma,double loss,double dd,int64_t version) {
    GUARD_BEGIN
    auto start=std::chrono::steady_clock::now();
    static_cast<Engine*>(p)->quote(ts,mid,spread,skew,band,size,sigma,max_sigma,loss,dd,version);
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
    GUARD_END
}
int64_t mm_flatten(void* p,int64_t ts) {
    GUARD_BEGIN auto* e=static_cast<Engine*>(p);e->cancel_all(false);
    if(e->inventory!=0) e->add(ts,e->next_id--,e->inventory>0?-1:1,1,std::abs(e->inventory),true,true,true);
    return 0; GUARD_END
}
}
