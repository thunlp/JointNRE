#include <cstring>
#include <cstdio>
#include <algorithm>

using namespace std;

struct arr{
	int num;
	float score;
	int tip;
	int gg;
};

int num, tip;
float score;

arr res[10000000];

struct cmp {
	bool operator()(const arr &a, const arr &b) {
		return (a.score > b.score)||(a.score == b.score && a.tip > b.tip);
	}
};

int main() {
	FILE *fin = fopen("res.txt", "r");
	int tot = 0;
	float sum = 0;
	while (fscanf(fin, "%d", &num) == 1) {
		fscanf(fin, "%f", &score);
		fscanf(fin, "%d", &tip);
		res[tot].score = score;
		res[tot].num = num;
		res[tot].tip = tip;
		res[tot].gg = tot;
		tot++;
		if (num != 51 && tip == 1) sum++;
	}
	fclose(fin);
	sort(res, res + tot, cmp());
	float acc = 0;
	float gg = 0;
	FILE *fout = fopen("log.txt", "w");
	for (int i = 0; i < tot; i++) {
		if (res[i].num == 51) continue;
		fprintf(fout, "%d %d %f %d %f %f\n", res[i].num, res[i].tip, res[i].score, res[i].gg/56, acc/sum, acc/gg);
		gg+=1;
		if (res[i].tip == 1) {
			acc++;
			printf("%f %f\n", acc/sum, acc/gg);
		}
	}
	fclose(fout);
	return 0;
}