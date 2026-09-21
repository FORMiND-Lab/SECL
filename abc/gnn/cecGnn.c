/* One explicit model invocation per command. Model scores never authorize merges. */
#include "cecGnn.h"
#include "cecGnnConfig.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

ABC_NAMESPACE_IMPL_START

static struct {
    const char *model, *checkpoint, *python, *script, *device;
    int active, prepass, nodes, dim, *proof;
    float *emb;
    double forward, normalize, bridge;
    long merges;
} gnn;

static double GnnTime(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + 1e-9 * t.tv_nsec;
}

static uint64_t GnnMix(uint64_t h, uint32_t x)
{
    int i;
    for (i = 0; i < 4; ++i) { h ^= (x >> (8*i)) & 255; h *= UINT64_C(1099511628211); }
    return h;
}

static uint64_t GnnFingerprint(Gia_Man_t *p)
{
    Gia_Obj_t *o;
    int i;
    uint64_t h = GnnMix(UINT64_C(1469598103934665603), Gia_ManObjNum(p));
    Gia_ManForEachObj(p, o, i) {
        h = GnnMix(h, Gia_ObjIsAnd(o) ? 1 : Gia_ObjIsCi(o) ? 0 : Gia_ObjIsCo(o) ? 2 : 3);
        if (Gia_ObjIsAnd(o) || Gia_ObjIsCo(o)) h = GnnMix(h, Gia_ObjFaninLit0p(p, o));
        if (Gia_ObjIsAnd(o)) h = GnnMix(h, Gia_ObjFaninLit1p(p, o));
    }
    return h;
}

int Cec_GnnEnabled(void) { return gnn.active; }
int Cec_GnnPrepass(void) { return gnn.prepass; }

int Cec_GnnInfer(Gia_Man_t *p)
{
    char directory[] = "/tmp/abc-gnn-XXXXXX", graph[128], output[128], magic[8];
    uint32_t nodes, dim;
    uint64_t fingerprint;
    Gia_Obj_t *o;
    FILE *f = NULL;
    int i, first = 1, status, ok = 0;
    pid_t child;
    double start = GnnTime();
    size_t count, k;
    if (p->nXors || p->pMuxes) {
        Abc_Print(-1, "GNN requires an AND/inverter graph; run &st first.\n");
        return 0;
    }
    if (!mkdtemp(directory)) { perror("GNN temporary directory"); return 0; }
    snprintf(graph, sizeof(graph), "%s/graph.json", directory);
    snprintf(output, sizeof(output), "%s/embeddings.bin", directory);
    f = fopen(graph, "w");
    if (!f) goto done;
    fprintf(f, "{\"nodes\":%d,\"fingerprint\":\"%016llx\",\"pis\":[",
            Gia_ManObjNum(p), (unsigned long long)GnnFingerprint(p));
    Gia_ManForEachCi(p, o, i) fprintf(f, "%s%d", i ? "," : "", Gia_ObjId(p, o));
    fprintf(f, "],\"ands\":[");
    Gia_ManForEachAnd(p, o, i) {
        fprintf(f, "%s[%d,%d,%d]", first ? "" : ",", i,
                Gia_ObjFaninLit0p(p, o), Gia_ObjFaninLit1p(p, o));
        first = 0;
    }
    fprintf(f, "]}\n");
    status = fclose(f); f = NULL;
    if (status) goto done;
    fflush(NULL);
    child = fork();
    if (child == 0) {
        execlp(gnn.python, gnn.python, gnn.script, "--model", gnn.model,
               "--checkpoint", gnn.checkpoint, "--device", gnn.device,
               "--graph", graph, "--output", output, (char *)NULL);
        perror("GNN Python execution");
        _exit(127);
    }
    if (child < 0) goto done;
    while (waitpid(child, &status, 0) < 0) if (errno != EINTR) goto done;
    if (!WIFEXITED(status) || WEXITSTATUS(status)) goto done;
    f = fopen(output, "rb");
    if (!f || fread(magic, 1, 8, f) != 8 || memcmp(magic, "ABCGNN1\0", 8) ||
        fread(&nodes, 4, 1, f) != 1 || fread(&dim, 4, 1, f) != 1 ||
        fread(&fingerprint, 8, 1, f) != 1 || fread(&gnn.forward, 8, 1, f) != 1 ||
        fread(&gnn.normalize, 8, 1, f) != 1) goto done;
    if (nodes != Gia_ManObjNum(p) || !dim || dim > 65536 ||
        fingerprint != GnnFingerprint(p) || !isfinite(gnn.forward) ||
        !isfinite(gnn.normalize) || gnn.forward < 0 || gnn.normalize < 0) goto done;
    count = (size_t)nodes * dim;
    gnn.emb = ABC_ALLOC(float, count);
    if (!gnn.emb || fread(gnn.emb, sizeof(float), count, f) != count || fgetc(f) != EOF) goto done;
    for (k = 0; k < count; ++k) if (!isfinite(gnn.emb[k])) goto done;
    gnn.nodes = nodes; gnn.dim = dim;
    ok = 1;
done:
    if (f) fclose(f);
    unlink(graph); unlink(output); rmdir(directory);
    gnn.bridge += GnnTime() - start;
    if (!ok) Abc_Print(-1, "GNN inference failed: command aborted.\n");
    return ok;
}

double Cec_GnnSimilarity(Gia_Man_t *p, int a, int b)
{
    double score = 0;
    int j;
    if (p->vIdsOrig) { a = Vec_IntEntry(p->vIdsOrig, a); b = Vec_IntEntry(p->vIdsOrig, b); }
    assert(a >= 0 && a < gnn.nodes && b >= 0 && b < gnn.nodes);
    for (j = 0; j < gnn.dim; ++j)
        score += (double)gnn.emb[(size_t)a * gnn.dim + j] * gnn.emb[(size_t)b * gnn.dim + j];
    return fmin(1.0, fabs(score));
}

void Cec_GnnRecordMerge(Gia_Man_t *p, int node, int repr)
{
    if (!gnn.prepass) return;
    assert(repr < node);
    gnn.proof[node] = Abc_Var2Lit(repr, Gia_ManObj(p, node)->fPhase ^ Gia_ManObj(p, repr)->fPhase);
    gnn.merges++;
}

Gia_Man_t *Cec_GnnReplay(Gia_Man_t *p)
{
    Gia_Man_t *q = Gia_ManStart(Gia_ManObjNum(p)), *clean;
    Gia_Obj_t *o;
    int i, a, b, proof, *copy = ABC_ALLOC(int, Gia_ManObjNum(p));
    q->pName = Abc_UtilStrsav(p->pName); q->pSpec = Abc_UtilStrsav(p->pSpec);
    copy[0] = 0;
    Gia_ManForEachCi(p, o, i) copy[Gia_ObjId(p, o)] = Gia_ManAppendCi(q);
    Gia_ManHashAlloc(q);
    Gia_ManForEachAnd(p, o, i) {
        proof = gnn.proof[i];
        if (proof >= 0) copy[i] = Abc_LitNotCond(copy[Abc_Lit2Var(proof)], Abc_LitIsCompl(proof));
        else {
            a = Abc_LitNotCond(copy[Gia_ObjFaninId0p(p, o)], Gia_ObjFaninC0(o));
            b = Abc_LitNotCond(copy[Gia_ObjFaninId1p(p, o)], Gia_ObjFaninC1(o));
            copy[i] = Gia_ManHashAnd(q, a, b);
        }
    }
    Gia_ManForEachCo(p, o, i) Gia_ManAppendCo(q, Abc_LitNotCond(copy[Gia_ObjFaninId0p(p, o)], Gia_ObjFaninC0(o)));
    Gia_ManSetRegNum(q, Gia_ManRegNum(p));
    Gia_ManHashStop(q);
    clean = Gia_ManCleanup(q); Gia_ManStop(q); ABC_FREE(copy);
    return clean;
}

int Cec_GnnRequested(int argc, char **argv)
{
    int i;
    for (i = 1; i < argc; ++i) if (!strcmp(argv[i], "-e")) return 1;
    return 0;
}

static void GnnFree(void)
{
    Cec_GnnBatchEnd();
    ABC_FREE(gnn.emb); ABC_FREE(gnn.proof);
    memset(&gnn, 0, sizeof(gnn));
}

int Cec_GnnCommand(Abc_Frame_t *frame, int argc, char **argv, int cec)
{
    const char *files[2];
    int i, nfiles = 0, conflicts = 100000, verbose = 0, miter = 0, result = -2;
    Gia_Man_t *p = NULL, *left = NULL, *right = NULL, *pre = NULL, *out = NULL;
    double start, algorithm, pretime = 0;
    GnnFree();
    gnn.python = "python3"; gnn.script = ABC_GNN_SCRIPT; gnn.device = "cpu";
    for (i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "-e") && i + 2 < argc && !gnn.model) {
            gnn.model = argv[++i]; gnn.checkpoint = argv[++i];
        } else if (!strcmp(argv[i], "-Y") && i + 1 < argc) gnn.python = argv[++i];
        else if (!strcmp(argv[i], "-G") && i + 1 < argc) gnn.script = argv[++i];
        else if (!strcmp(argv[i], "-U") && i + 1 < argc) gnn.device = argv[++i];
        else if (!strcmp(argv[i], "-C") && i + 1 < argc) {
            char *end;
            long value = strtol(argv[++i], &end, 10);
            if (!argv[i][0] || *end || value < 1 || value > 2147483647) goto usage;
            conflicts = value;
        } else if (!strcmp(argv[i], "-v")) verbose = 1;
        else if (cec && !strcmp(argv[i], "-m")) miter = 1;
        else if (cec && argv[i][0] != '-' && nfiles < 2) files[nfiles++] = argv[i];
        else goto usage;
    }
    if (!gnn.model || strcmp(gnn.model, "DeepGate2")) goto usage;
    if (access(gnn.checkpoint, R_OK) || access(gnn.script, R_OK)) {
        Abc_Print(-1, "GNN checkpoint or inference script is not readable: %s ; %s\n", gnn.checkpoint, gnn.script);
        goto done;
    }
    if (cec) {
        if (miter) {
            Gia_Obj_t *o;
            if (nfiles || !Abc_FrameReadGia(frame)) goto usage;
            left = Gia_ManDup(Abc_FrameReadGia(frame));
            right = Gia_ManStart(Gia_ManCiNum(left) + Gia_ManCoNum(left) + 1);
            Gia_ManForEachCi(left, o, i) Gia_ManAppendCi(right);
            Gia_ManForEachCo(left, o, i) Gia_ManAppendCo(right, 0);
        } else {
            if (nfiles != 2) goto usage;
            for (i = 0; i < 2; ++i) if (access(files[i], R_OK)) {
                Abc_Print(-1, "Cannot read input AIG: %s\n", files[i]); goto done;
            }
            left = Gia_AigerRead((char *)files[0], 0, 0, 1);
            right = Gia_AigerRead((char *)files[1], 0, 0, 1);
            if (!left || !right) goto done;
        }
        if (Gia_ManRegNum(left) || Gia_ManRegNum(right) ||
            Gia_ManCiNum(left) != Gia_ManCiNum(right) || Gia_ManCoNum(left) != Gia_ManCoNum(right)) {
            Abc_Print(-1, "GNN CEC requires combinational AIGs with matching PI/PO counts.\n"); goto done;
        }
        p = Gia_ManMiter(left, right, 0, 1, 0, 0, verbose);
        if (!p) goto done;
        gnn.active = 1;
        start = GnnTime();
        {
            Cec_ParCec_t pars;
            Cec_ManCecSetDefaultParams(&pars);
            pars.nBTLimit = conflicts; pars.fVerbose = verbose;
            result = Cec_ManVerify(p, &pars);
        }
        algorithm = GnnTime() - start - gnn.bridge;
        if (result == -2) goto done;
        Abc_FrameSetStatus(result);
        Abc_FrameReplaceCex(frame, &p->pCexComb);
        printf("[GNN_CEC] status=%d abc_seconds=%.9f\n", result, algorithm);
    } else {
        Cec_ParFra_t pars;
        if (!Abc_FrameReadGia(frame)) { Abc_Print(-1, "No current AIG.\n"); goto done; }
        // Work on a copy so an inference error leaves the current network intact.
        p = Gia_ManDup(Abc_FrameReadGia(frame));
        if (!Cec_GnnInfer(p)) goto done;
        start = GnnTime();
        gnn.proof = ABC_FALLOC(int, Gia_ManObjNum(p));
        gnn.prepass = 1;
        Cec4_ManSetParams(&pars);
        pars.nBTLimit = Abc_MinInt(conflicts, 1000); pars.fVerbose = verbose;
        pre = Cec4_ManSimulateTest(p, &pars);
        gnn.prepass = 0;
        pretime = GnnTime() - start;
        Cec4_ManSetParams(&pars);
        pars.nBTLimit = conflicts; pars.nGenIters = 0; pars.fVerbose = verbose;
        out = Cec4_ManSimulateTest(pre, &pars);
        algorithm = GnnTime() - start;
        Abc_FrameUpdateGia(frame, out); out = NULL;
        result = 1;
        printf("[GNN_FRAIG] prepass_seconds=%.9f cleanup_seconds=%.9f merges=%ld cap=128 probe=%d\n",
               pretime, algorithm - pretime, gnn.merges, Abc_MinInt(conflicts, 1000));
    }
    printf("[GNN_TIME] forward_seconds=%.9f normalization_seconds=%.9f inference_seconds=%.9f abc_seconds=%.9f algorithm_seconds=%.9f cold_start_included=0\n",
           gnn.forward, gnn.normalize, gnn.forward + gnn.normalize, algorithm,
           algorithm + gnn.forward + gnn.normalize);
done:
    if (p) Gia_ManStop(p);
    if (pre) Gia_ManStop(pre);
    if (left) Gia_ManStop(left);
    if (right) Gia_ManStop(right);
    GnnFree();
    return result == -2 ? 1 : 0;
usage:
    Abc_Print(-1, "usage: %s -e MODEL CHECKPOINT [-Y python] [-G infer.py] [-U cpu|cuda:N] [-C conflicts] [-v]%s\n",
              argv[0], cec ? " LEFT.aig RIGHT.aig (or -m for current miter)" : "");
    Abc_Print(-1, "Model: DeepGate2. Missing/invalid options abort the command.\n");
    goto done;
}

ABC_NAMESPACE_IMPL_END
