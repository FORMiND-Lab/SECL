/* SECL candidate ordering for the synthesis prepass and CEC. */
#include "cecGnn.h"
#include <math.h>

ABC_NAMESPACE_IMPL_START
typedef struct { double score; int node; } GnnRank;
static struct {
    Cec_ManFra_t *fra;
    Cec_ManSim_t *sim;
    Cec_ManPat_t *pat;
    int *order, count, pending;
    unsigned char *skip;
} batch;

static int GnnCompare(const void *a, const void *b)
{
    const GnnRank *x = a, *y = b;
    if (x->score != y->score) return x->score > y->score ? -1 : 1;
    return (x->node > y->node) - (x->node < y->node);
}

Vec_Int_t *Cec_GnnPrepassOrder(Gia_Man_t *p)
{
    Gia_Obj_t *o;
    int i, repr, count = 0, own = p->vFanoutNums == NULL;
    GnnRank *rank = ABC_ALLOC(GnnRank, Gia_ManAndNum(p));
    Vec_Int_t *order = Vec_IntAlloc(128);
    if (own) Gia_ManStaticFanoutStart(p);
    Gia_ManLevelNum(p);
    Gia_ManForEachAnd(p, o, i) {
        double sim, score;
        repr = Gia_ObjRepr(p, i);
        if (repr == GIA_VOID) continue;
        sim = Cec_GnnSimilarity(p, i, repr);
        score = pow(sim, 8.0) * (1 + Gia_ObjFanoutNumId(p, i) + Gia_ObjFanoutNumId(p, repr))
              / sqrt(1.0 + Gia_ObjLevelId(p, i));
        rank[count++] = (GnnRank){score, i};
    }
    if (own) Gia_ManStaticFanoutStop(p);
    qsort(rank, count, sizeof(GnnRank), GnnCompare);
    for (i = 0; i < Abc_MinInt(128, count); ++i) Vec_IntPush(order, rank[i].node);
    ABC_FREE(rank);
    return order;
}

static void GnnTfo(Gia_Man_t *p, int node, Vec_Int_t *nodes)
{
    int i, fan;
    if (Gia_ObjIsTravIdCurrentId(p, node)) return;
    Gia_ObjSetTravIdCurrentId(p, node);
    if (!Gia_ObjIsAnd(Gia_ManObj(p, node))) return;
    Gia_ObjForEachFanoutStaticId(p, node, fan, i) GnnTfo(p, fan, nodes);
    Vec_IntPush(nodes, node);
}

void Cec_GnnBatchBegin(Cec_ManFra_t *fra, Cec_ManSim_t *sim, Cec_ManPat_t *pat, Gia_Man_t *srm)
{
    Gia_Man_t *p = fra->pAig;
    Gia_Obj_t *o;
    int k, i, node, repr, own;
    double *similarity;
    GnnRank *rank;
    Vec_Int_t *roots, *nodes;
    Cec_GnnBatchEnd();
    if (!Cec_GnnEnabled()) return;
    assert(fra->pPars->nDepthMax == 1);
    batch.fra = fra; batch.sim = sim; batch.pat = pat;
    batch.count = Gia_ManCoNum(srm);
    batch.order = ABC_ALLOC(int, batch.count);
    batch.skip = ABC_CALLOC(unsigned char, batch.count);
    rank = ABC_ALLOC(GnnRank, batch.count);
    similarity = ABC_CALLOC(double, Gia_ManObjNum(p));
    roots = Vec_IntAlloc(2); nodes = Vec_IntAlloc(1000);
    Gia_ManForEachAnd(p, o, node) {
        repr = Gia_ObjRepr(p, node);
        if (repr != GIA_VOID) similarity[node] = Cec_GnnSimilarity(p, node, repr);
    }
    own = p->vFanoutNums == NULL;
    if (own) Gia_ManStaticFanoutStart(p);
    // Score only current query targets; TFO sums use all current candidates.
    for (k = 0; k < batch.count; ++k) {
        int tfi = 0;
        double tfo = 0;
        repr = Vec_IntEntry(fra->vXorNodes, 2*k);
        node = Vec_IntEntry(fra->vXorNodes, 2*k+1);
        Vec_IntClear(roots);
        if (repr > 0) Vec_IntPush(roots, repr);
        Vec_IntPush(roots, node);
        Gia_ManCollectTfi(p, roots, nodes);
        Vec_IntForEachEntry(nodes, i, node)
            tfi += Gia_ObjRepr(p, i) != GIA_VOID && !Gia_ObjProved(p, i);
        Vec_IntClear(nodes); Gia_ManIncrementTravId(p);
        Vec_IntForEachEntry(roots, i, node) GnnTfo(p, i, nodes);
        Vec_IntForEachEntry(nodes, i, node) if (!Gia_ObjProved(p, i)) tfo += similarity[i];
        node = Vec_IntEntry(fra->vXorNodes, 2*k+1);
        rank[k] = (GnnRank){similarity[node] * (tfi + tfo), node};
    }
    if (own) Gia_ManStaticFanoutStop(p);
    qsort(rank, batch.count, sizeof(GnnRank), GnnCompare);
    {
        int *query = ABC_FALLOC(int, Gia_ManObjNum(p));
        for (k = 0; k < batch.count; ++k) query[Vec_IntEntry(fra->vXorNodes, 2*k+1)] = k;
        for (k = 0; k < batch.count; ++k) batch.order[k] = query[rank[k].node];
        ABC_FREE(query);
    }
    Vec_IntFree(roots); Vec_IntFree(nodes); ABC_FREE(rank); ABC_FREE(similarity);
}

int Cec_GnnOrder(int k) { return batch.order ? batch.order[k] : k; }
int Cec_GnnSkipped(int k) { return batch.skip && batch.skip[k]; }

int Cec_GnnBefore(int k)
{
    int repr, node;
    if (!batch.order) return 0;
    repr = Vec_IntEntry(batch.fra->vXorNodes, 2*k);
    node = Vec_IntEntry(batch.fra->vXorNodes, 2*k+1);
    if (batch.sim->pCexComb || Gia_ObjRepr(batch.fra->pAig, node) != repr) {
        batch.skip[k] = 1;
        return 1;
    }
    return 0;
}

void Cec_GnnAfter(int k, int status)
{
    Vec_Ptr_t *info;
    int series, i, w;
    if (!batch.order || status != 0 || ++batch.pending < 32) return;
    info = Cec_ManPatCollectPatterns(batch.pat, Gia_ManCiNum(batch.fra->pAig), batch.sim->nWords);
    Gia_ManCreateValueRefs(batch.fra->pAig);
    for (series = 0; series < batch.pat->nSeries; ++series) {
        for (i = 0; i < Gia_ManCiNum(batch.fra->pAig); ++i) {
            unsigned *dst = Vec_PtrEntry(batch.sim->vCiSimInfo, i);
            unsigned *src = (unsigned *)Vec_PtrEntry(info, i) + batch.sim->nWords * series;
            for (w = 0; w < batch.sim->nWords; ++w) dst[w] = src[w];
        }
        if (Cec_ManSimSimulateRound(batch.sim, batch.sim->vCiSimInfo, batch.sim->vCoSimInfo)) break;
    }
    Vec_PtrFree(info);
    batch.pat->iStart = Vec_StrSize(batch.pat->vStorage);
    batch.pending = 0;
}

void Cec_GnnBatchEnd(void)
{
    ABC_FREE(batch.order); ABC_FREE(batch.skip);
    memset(&batch, 0, sizeof(batch));
}
ABC_NAMESPACE_IMPL_END
