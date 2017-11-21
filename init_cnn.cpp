#include <cstring>
#include <cstdio>
#include <string>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

string path = "./data/FB60K/";

vector<int> Nalist, FNalist;
int NA = -1;
const float pi = 3.141592653589793238462643383;
int rate = 3;

float rand(float min, float max) {
	return min + (max - min) * rand() / (RAND_MAX + 1.0);
}

float normal(float x, float miu,float sigma) {
	return 1.0/sqrt(2*pi)/sigma*exp(-1*(x-miu)*(x-miu)/(2*sigma*sigma));
}

float randn(float miu,float sigma, float min ,float max) {
	float x, y, dScope;
	do {
		x = rand(min,max);
		y = normal(x,miu,sigma);
		dScope=rand(0.0,normal(miu,miu,sigma));
	} while (dScope > y);
	return x;
}

int word_size;
int entity_size;
int dimension;
int PositionLimit = 30;
int LenLimit = 100;
float *word_embeddings;

struct Tip {
	int h;
	int t;
	int r;
	int tot;
	int *lists;
};

Tip *tipList;
int tipTotal;
int midTotal;
int relationTotal = 0;
int sentenceTotal;
int instanceTot = 0;
int *sentence, *posH, *posT, *bags_train;

int getPosition(int position) {
	if (position < -PositionLimit) return 0;
	if (position > PositionLimit) return 2 * PositionLimit;
	return position + PositionLimit;
}

extern "C"
void setNA(int con) {
	NA = con;
}

extern "C"
void setRate(int con) {
	rate = con;
}

extern "C"
void readWordVec() {

	FILE *fin1 = fopen((path + "entity2id.txt").c_str(), "r");
	FILE *fin2 = fopen((path + "vec.txt").c_str(), "r");
	fscanf(fin2, "%d%d\n", &word_size, &dimension);
	fscanf(fin1, "%d", &entity_size);

	char buffer[100];
	word_size += 2 + entity_size;
	word_embeddings = (float *)calloc(word_size * dimension, sizeof(float));
	int last = 0;
	for (int i = entity_size * dimension; i > 0 ; i--)
		word_embeddings[last++] = randn(0, 1.0 / dimension, -6 / sqrt(dimension), 6 / sqrt(dimension));
	last += 2 * dimension;
	for (int i = entity_size + 2; i < word_size; i++) {
		fscanf(fin2, "%s", buffer);
		for (int i = 0; i < dimension; i++)
			fscanf(fin2, "%f", &word_embeddings[last++]);
	}

	fclose(fin2);
	fclose(fin1);
}

extern "C"
void getWordVec(float *con) {
	for (int i = word_size * dimension - 1; i >= 0; i--)
		con[i] = word_embeddings[i];
}

int n = 0;
int lastNA = 0;
int lastFNA = 0;
extern "C"
int batch_iter(int *x_batch, int *p_h_batch, int *p_t_batch, int *y_batch, int *r_batch, float *r_n_batch, int *h_batch, int *t_batch) {
	 if (rate == 0 || Nalist.size() == 0) {
	 	n = FNalist[lastFNA];
	 	r_n_batch[0] = 1;
	 	lastFNA = (lastFNA + 1) % (FNalist.size());
	 } else if (rand() % rate != 0) {
	 	n = Nalist[lastNA];
	 	r_n_batch[0] = 0;
	 	lastNA = (lastNA + 1) % (Nalist.size());
	 } else {
	 	n = FNalist[lastFNA];
	 	r_n_batch[0] = 1;
	 	lastFNA = (lastFNA + 1) % (FNalist.size());
	 }
	int instance = tipList[n].tot;
	int last = 0;
	for (int i = 0; i < instance; i++) {
		int j = tipList[n].lists[i];
		int last1 = j * LenLimit;
		for (int k = 0; k < LenLimit; k++) {
			x_batch[last] = sentence[last1];
			p_h_batch[last] = posH[last1];
			p_t_batch[last] = posT[last1];
			last++;
			last1++;
		}
	}
	r_batch[0] = tipList[n].r;
	for (int j = 0; j < relationTotal; j++)
		y_batch[j] = 0;
	y_batch[tipList[n].r] = 1;
	h_batch[0] = tipList[n].h;
	t_batch[0] = tipList[n].t;
	return instance;
}

extern "C"
int getTipTotal() {
	return tipTotal;
}

extern "C"
int getLenLimit() {
	return LenLimit;
}

extern "C"
int getRelationTotal() {
	return relationTotal;
}

extern "C"
int getWordTotal() {
	return word_size;
}

extern "C"
int getPositionLimit() {
	return PositionLimit;
}

extern "C"
int getWordDimension() {
	return dimension;
}

extern "C"
int getInstanceTot() {
	return instanceTot;
}

extern "C"
void readFromFile() {
	FILE *f = fopen((path + "corpus2id.txt").c_str(), "r");
	fscanf(f, "%d\n", &tipTotal);
	fscanf(f, "%d\n", &sentenceTotal);
	tipList = (Tip *)calloc(tipTotal, sizeof(Tip));
	sentence = (int *)calloc(sentenceTotal * LenLimit, sizeof(int));
	posH = (int *)calloc(sentenceTotal * LenLimit, sizeof(int));
	posT = (int *)calloc(sentenceTotal * LenLimit, sizeof(int));
	bags_train = (int *)calloc(sentenceTotal, sizeof(int));
	int h, t, post, posh, r, len, tip;
	for (int i = 0; i < sentenceTotal; i++) {
		fscanf(f, "%d%d%d%d%d%d%d",&h, &t, &posh, &post, &r, &tip, &len);
		int last = i * LenLimit;
		for (int j = 0; j < len; j++) {
			fscanf(f, "%d", &sentence[last + j]);
			posH[last + j] = getPosition(j - posh);
			posT[last + j] = getPosition(j - post);
		}
		bags_train[i] = tip;
		tipList[tip].tot++;
		tipList[tip].h = h;
		tipList[tip].r = r;
		tipList[tip].t = t;
		if (r + 1 > relationTotal)
			relationTotal = r + 1;
		if (tipList[tip].tot > instanceTot)
			instanceTot = tipList[tip].tot;
	}
	fclose(f);
	for (int i = 0; i < tipTotal; i++) {
		tipList[i].lists = new int[tipList[i].tot];
		tipList[i].tot = 0;
		if (tipList[i].r == NA) Nalist.push_back(i); else FNalist.push_back(i);
	}
	for (int i = 0; i < sentenceTotal; i++)
		tipList[bags_train[i]].lists[tipList[bags_train[i]].tot++] = i;
}

int main() {
	readWordVec();
	readFromFile();
	return 0;
}
