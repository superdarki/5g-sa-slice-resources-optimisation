#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <unistd.h>
struct res_sim
{
    double loss;
    double wait_avg;
    double wait_max;
    double urllc_tot;
    double urllc_max;
    double embb_tot;
};

void transition(double lambda_e, double lambda_u, double mu_e, double mu_u, int S, int G, int x1, int x2, int x3, double *duree, int etat[3])
{
    int etats[5][3];
    double taux[5];
    int count = 0;

    if (x1 > 0)
    {
        etats[count][0] = x1 - 1;
        etats[count][1] = x2;
        etats[count][2] = x3;
        taux[count] = mu_u * x1;
        count++;
    }

    if (x2 > 0)
    {
        if ((x1 + x2 <= S - G) && x3 > 0)
        {
            etats[count][0] = x1;
            etats[count][1] = x2;
            etats[count][2] = x3 - 1;
        }
        else
        {
            etats[count][0] = x1;
            etats[count][1] = x2 - 1;
            etats[count][2] = x3;
        }
        taux[count] = mu_e * x2;
        count++;
    }

    if (x1 + x2 < S)
    {
        etats[count][0] = x1 + 1;
        etats[count][1] = x2;
        etats[count][2] = x3;
        taux[count] = lambda_u;
        count++;
    }

    if (x1 + x2 < S - G)
    {
        etats[count][0] = x1;
        etats[count][1] = x2 + 1;
        etats[count][2] = x3;
        taux[count] = lambda_e;
    }
    else
    {
        etats[count][0] = x1;
        etats[count][1] = x2;
        etats[count][2] = x3 + 1;
        taux[count] = lambda_e;
    }
    count++;

    double param_expo = 0.0;
    for (int i = 0; i < count; i++)
    {
        param_expo += taux[i];
    }

    *duree = -1.0 / param_expo * log((double)rand() / RAND_MAX);

    double cumulative_sum = 0.0;
    double u = (double)rand() / RAND_MAX;
    int index = 0;
    for (int i = 0; i < count; i++)
    {
        cumulative_sum += taux[i] / param_expo;
        if (u <= cumulative_sum)
        {
            index = i;
            break;
        }
    }

    etat[0] = etats[index][0];
    etat[1] = etats[index][1];
    etat[2] = etats[index][2];
}

void simu(double lambda_e, double lambda_u, double mu_e, double mu_u, int S, int G, int NbIter, struct res_sim *res)
{
    int e[3] = {0, 0, 0};
    double cumul = 0.0;
    double temps_total = 0.0;
    double horizon = (double)NbIter / (lambda_e + lambda_u);
    double t = 0.0;
    double wait_avg = 0.0;
    double wait_max = 0.0;
    double urllc_tot = 0.0;
    double urllc_max = 0.0;
    double embb_tot = 0.0;

    srand(time(NULL) ^ getpid());

    while (temps_total < horizon)
    {
        t = 0.0;
        int e_new[3] = {0, 0, 0};
        transition(lambda_e, lambda_u, mu_e, mu_u, S, G, e[0], e[1], e[2], &t, e_new);
        temps_total += t;
        wait_avg += e[2] * t;
        wait_max = (wait_max > e[2]) ? wait_max : e[2];
        if (e_new[0] > e[0])
        {
            urllc_tot += 1;
        }
        if (e_new[0] > urllc_max)
        {
            urllc_max = e_new[0];
        }
        if (e_new[1] > e[1])
        {
            embb_tot += 1;
        }
        if (e[0] + e[1] == S)
        {
            cumul += t;
        }
        e[0] = e_new[0];
        e[1] = e_new[1];
        e[2] = e_new[2];
    }

    if (e[0] + e[1] == S)
    {
        cumul += t - (temps_total - horizon);
    }

    res->loss = cumul / horizon;
    res->wait_avg = wait_avg / horizon;
    res->wait_max = wait_max;
    res->urllc_tot = urllc_tot;
    res->urllc_max = urllc_max;
    res->embb_tot = embb_tot;
}

/**
 * ============================================================================
 * ./simulation <lambda_e> <lambda_u> <mu_e> <mu_u> <S> <G> <NbIter>
 *
 * Example:
 * ./simulation 10.0 5.0 2.0 10.0 10 2 100000
 * ============================================================================
 */
int main(int argc, char *argv[])
{
    // --- 1. Argument Parsing and Validation ---

    if (argc != 8)
    {
        fprintf(stderr, "Usage: %s <lambda_e> <lambda_u> <mu_e> <mu_u> <S> <G> <NbIter>\n", argv[0]);
        fprintf(stderr, "\n  <lambda_e>: eMBB arrival rate (packets/sec)\n");
        fprintf(stderr, "  <lambda_u>: URLLC arrival rate (packets/sec)\n");
        fprintf(stderr, "  <mu_e>:     eMBB service rate (packets/sec)\n");
        fprintf(stderr, "  <mu_u>:     URLLC service rate (packets/sec)\n");
        fprintf(stderr, "  <S>:        Total number of resources (channels)\n");
        fprintf(stderr, "  <G>:        Guard channels for URLLC\n");
        fprintf(stderr, "  <NbIter>:   Number of iterations for time horizon\n\n");
        fprintf(stderr, "Example: %s 10.0 5.0 2.0 10.0 10 2 100000\n", argv[0]);
        return EXIT_FAILURE;
    }

    double lambda_e = atof(argv[1]);
    double lambda_u = atof(argv[2]);
    double mu_e = atof(argv[3]);
    double mu_u = atof(argv[4]);
    int S = atoi(argv[5]);
    int G = atoi(argv[6]);
    int NbIter = atoi(argv[7]);

    // Basic validation
    if (lambda_e < 0 || lambda_u < 0 || mu_e < 0 || mu_u < 0 || S < 0 || G < 0 || NbIter <= 0)
    {
        fprintf(stderr, "Error: All rates, counts, and iterations must be non-negative.\n");
        return EXIT_FAILURE;
    }
    if (G > S)
    {
        fprintf(stderr, "Error: Guard channels (G) cannot be greater than total resources (S).\n");
        return EXIT_FAILURE;
    }

    // --- 2. Simulation Setup and Execution ---

    struct res_sim results;

    printf("Starting simulation with the following parameters:\n");
    printf("--------------------------------------------------\n");
    printf("  eMBB Arrival Rate (lambda_e): %.2f\n", lambda_e);
    printf("  URLLC Arrival Rate (lambda_u):%.2f\n", lambda_u);
    printf("  eMBB Service Rate (mu_e):     %.2f\n", mu_e);
    printf("  URLLC Service Rate (mu_u):    %.2f\n", mu_u);
    printf("  Total Resources (S):          %d\n", S);
    printf("  Guard Channels (G):           %d\n", G);
    printf("  Simulation Iterations:        %d\n", NbIter);
    printf("--------------------------------------------------\n\n");

    simu(lambda_e, lambda_u, mu_e, mu_u, S, G, NbIter, &results);

    // --- 3. Displaying Results ---

    printf("--- Simulation Results ---\n");
    printf("URLLC Packet Loss Ratio:     %e\n", results.loss);
    printf("Total URLLC Packets Arrived: %.0f\n", results.urllc_tot);
    printf("Max URLLC number in system : %.0f\n", results.urllc_max);
    printf("Total eMBB Packets Arrived:  %.0f\n", results.embb_tot);
    printf("Avg. eMBB Queue Size:        %f\n", results.wait_avg);
    printf("Final eMBB Queue Size:       %.0f\n", results.wait_max);
    printf("--------------------------\n");

    return EXIT_SUCCESS;
}