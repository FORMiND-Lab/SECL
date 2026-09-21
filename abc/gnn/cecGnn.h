#ifndef ABC_CEC_GNN_H
#define ABC_CEC_GNN_H
#include "cecInt.h"
#include "base/main/main.h"

ABC_NAMESPACE_HEADER_START
extern void Cec4_ManSetParams(Cec_ParFra_t *pars);
extern Gia_Man_t *Cec4_ManSimulateTest(Gia_Man_t *p, Cec_ParFra_t *pars);
extern int Cec_GnnRequested(int argc, char **argv);
extern int Cec_GnnCommand(Abc_Frame_t *frame, int argc, char **argv, int cec);
extern int Cec_GnnEnabled(void);
extern int Cec_GnnInfer(Gia_Man_t *p);
extern double Cec_GnnSimilarity(Gia_Man_t *p, int a, int b);
extern int Cec_GnnPrepass(void);
extern void Cec_GnnRecordMerge(Gia_Man_t *p, int node, int repr);
extern Gia_Man_t *Cec_GnnReplay(Gia_Man_t *p);
extern Vec_Int_t *Cec_GnnPrepassOrder(Gia_Man_t *p);
extern void Cec_GnnBatchBegin(Cec_ManFra_t *p, Cec_ManSim_t *sim,
                              Cec_ManPat_t *pat, Gia_Man_t *srm);
extern int Cec_GnnOrder(int k);
extern int Cec_GnnBefore(int k);
extern void Cec_GnnAfter(int k, int status);
extern int Cec_GnnSkipped(int k);
extern void Cec_GnnBatchEnd(void);
ABC_NAMESPACE_HEADER_END
#endif
