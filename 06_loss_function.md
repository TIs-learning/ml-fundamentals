# Modul Pembelajaran: Loss Functions (Optimization Target)

Modul pembelajaran ini membedah konsep **Fungsi Kerugian (*Loss Function*)** dalam *Machine Learning* (ML)—kompas matematika yang memandu model untuk mengetahui seberapa buruk tebakannya, seberapa besar hukuman yang harus diterima, dan ke arah mana parameter harus diperbaiki agar model menjadi semakin cerdas.

---

## Glosarium Istilah Penting
Sebelum mendalami materi, pahami pengertian dari istilah-istilah kunci berikut:
* **Fungsi Kerugian (*Loss Function*)**: rumus matematika yang menghitung seberapa jauh hasil tebakan model menyimpang dari target yang sebenarnya untuk satu data.
* **Galat (*Error*)**: selisih matematis murni antara nilai fakta target dengan nilai hasil tebakan ($y - \hat{y}$).
* **Metrik (*Metric*)**: angka tolok ukur yang mudah dipahami manusia untuk menilai kualitas performa model secara keseluruhan (misalnya: persentase akurasi).
* **Dapat Diturunkan (*Differentiable*)**: sifat fungsi matematika yang memiliki kurva mulus tanpa patahan tajam sehingga nilai kemiringannya (gradien) dapat dihitung menggunakan kalkulus.
* **Gradien (*Gradient*)**: arah dan kecuraman lereng matematika yang menunjukkan ke mana bobot model harus digeser agar kesalahan tebakan mengecil.
* **Pencilan (*Outlier*)**: data anomali yang nilainya melompat sangat jauh dan berbeda ekstrem dari rata-rata sebaran data normal.

---

## 1. Apa Sebenarnya Fungsi Kerugian Itu? (*What is a Loss Function?*)

Ketika sebuah model *Machine Learning* baru pertama kali dibuat, seluruh parameter bobot di dalamnya biasanya berupa angka acak. Akibatnya, tebakan pertamanya pasti sangat buruk dan ngawur.

Bagaimana komputer tahu bahwa tebakannya salah? Komputer tidak memiliki perasaan bersalah atau akal sehat. Komputer membutuhkan sebuah **skor hukuman numerik** yang pasti.

### 1.1 Analogi Permainan Panas-Dingin
Bayangkan kamu bermain tebak-tebakan mencari benda tersembunyi dengan mata tertutup:
* Setiap kali kamu melangkah, temanmu berkata: *“Makin dingin!”* (artinya kamu makin menjauh dari target) atau *“Makin panas!”* (artinya kamu makin mendekat).
* **Fungsi kerugian adalah teman pemandu tersebut**: sebuah fungsi matematika yang memberikan sinyal seberapa jauh jarak tebakan model saat ini dari posisi target yang sebenarnya.

```
+-----------------------------------------------------------------------------+
|                     PERAN LOSS FUNCTION DALAM LATIHAN                       |
+-----------------------------------------------------------------------------+

  Data Masukan (X) ---> [ MODEL ML ] ---> Hasil Tebakan (y_pred)
                             ^                     |
                             |                     v
                      Perbaiki Bobot      [ LOSS FUNCTION ] <--- Target Sejati (y)
                      lewat Gradien                |
                             |                     v
                             +------------- Nilai Hukuman (Loss)
```

### 1.2 Definisi Formal Matematis
Secara matematis, untuk sebuah sampel data ke-$i$, fungsi kerugian dinyatakan sebagai:

$$L(y_i, \hat{y}_i)$$

* $y_i$ adalah **target yang ingin diprediksi** (fakta lapangan sejati).
* $\hat{y}_i$ adalah **hasil tebakan model**.
* $L$ adalah fungsi yang memetakan perbedaan kedua nilai tersebut menjadi angka skalar riil positif ($\ge 0$).

Jika tebakan model tepat sempurna ($\hat{y} = y$), nilai kerugian bernilai **nol**. Semakin ngawur tebakannya, nilai kerugian akan **semakin besar**. Tujuan latihan adalah memaksa nilai ini turun serendah mungkin mendekati nol.

---

### 1.3 Sumber Rujukan Topik 1
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 5: “Machine Learning Basics”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 4: “Training Models”.

---

## 2. Komparasi: Loss vs Error vs Metric

Tiga istilah ini sering dipakai bergantian sehingga menimbulkan kerancuan bagi pemula, padahal perannya sangat berbeda.

```
                  PETA PERBEDAAN: ERROR vs LOSS vs METRIC
                                     |
        +----------------------------+----------------------------+
        |                            |                            |
        v                            v                            v
   GALAT (ERROR)           FUNGSI KERUGIAN (LOSS)          METRIK (METRIC)
  - Selisih mentah.       - Rumus hukuman optimasi.       - Evaluasi manusia/bisnis.
  - Sederhana: y - y_pred - Wajib bisa diturunkan.        - Sering kali tidak mulus.
  - Untuk 1 sampel data.  - Dipakai oleh komputer.        - Dipakai oleh stakeholder.
```

---

### 2.1 Rincian Karakteristik

#### A. Galat (*Error*)
* **Pengertian**: selisih pengurangan mentah antara target nyata dengan hasil tebakan ($e = y - \hat{y}$).
* **Karakteristik**: bisa bernilai positif (tebakan terlalu rendah) atau bernilai negatif (tebakan terlalu tinggi). Karena bisa saling menghilangkan jika dijumlahkan, galat mentah tidak bisa langsung dipakai untuk melatih model.

#### B. Fungsi Kerugian (*Loss / Cost Function*)
* **Pengertian**: transformasi matematis dari galat menjadi nilai tunggal positif yang dirancang khusus agar **dapat diturunkan (*differentiable*)**.
* **Pengguna Utama**: **algoritma optimasi komputer** (seperti *Gradient Descent*).
* Komputer membaca nilai *loss* untuk menghitung turunan kalkulus dan menentukan ke mana arah perbaikan parameter dalam (*bobot*).

#### C. Metrik (*Evaluation Metric*)
* **Pengertian**: nilai tolok ukur akhir yang digunakan oleh **manusia (perekayasa dan manajer bisnis)** untuk menilai seberapa bagus model bekerja.
* **Karakteristik**: metrik tidak wajib memiliki sifat turunan kalkulus yang mulus. Metrik berfokus pada kemudahan interpretasi manusia (misalnya: Akurasi $94\%$, Presisi $88\%$, atau skor kepuasan rekomendasi).

---

### 2.2 Tabel Komparasi Rinci
| Dimensi Evaluasi | Galat (*Error*) | Fungsi Kerugian (*Loss*) | Metrik (*Metric*) |
| :--- | :--- | :--- | :--- |
| **Definisi Dasar** | Selisih nilai murni ($y - \hat{y}$) | Nilai penalti yang dioptimalkan | Skor evaluasi kualitas sistem |
| **Konsumen Utama** | Komponen dasar hitungan | Algoritma optimasi (komputer) | Perekayasa dan bisnis (manusia) |
| **Sifat Matematis** | Tanda plus/minus biasa | Wajib kontinu dan dapat diturunkan | Bebas (boleh fungsi tangga/diskrit) |
| **Orientasi Nilai** | Mendekati angka nol | Selalu dicari nilai minimum | Tergantung metrik (bisa dimaksimalkan) |
| **Contoh Nyata** | $500.000 - 450.000 = 50.000$ | *Binary Cross-Entropy*, MSE | Akurasi, F1-Score, ROC-AUC, MAPE |

---

### 2.3 Sumber Rujukan Topik 2
* **Chollet, François (2021)**, *Deep Learning with Python*, Edisi ke-2, Manning Publications, Bab 3: “Introduction to Keras and TensorFlow”.
* **Murphy, Kevin P. (2012)**, *Machine Learning: A Probabilistic Perspective*, MIT Press, Bab 5: “Bayesian Statistics (Loss Functions)”.

---

## 3. Regression Losses: MAE vs MSE (Intuisi Visual)

Pada kasus regresi (menebak angka kontinu, seperti harga rumah atau temperatur), ada dua fungsi kerugian yang paling populer: **MAE** dan **MSE**.

---

### 3.1 Mean Absolute Error (MAE / L1 Loss)

#### A. Konsep dan Formula
MAE mengukur **jarak absolut rata-rata** antara target sebenarnya dengan hasil tebakan tanpa memedulikan arah plus atau minusnya:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$

#### B. Intuisi Hukuman
MAE memperlakukan kesalahan secara **adil dan proporsional (garis lurus)**:
* Jika tebakanmu meleset sebesar $2$ poin, hukumannya bernilai $2$.
* Jika tebakanmu meleset sebesar $10$ poin, hukumannya bernilai $10$ (tepat lima kali lipat).

#### C. Keunggulan Utama: Kebal Terhadap Pencilan (*Robust to Outliers*)
Karena hukumannya bertambah secara linear, MAE tidak terlalu panik ketika bertemu satu atau dua data anomali yang angkanya melompat sangat tinggi. MAE cocok digunakan jika dataset kamu banyak mengandung derau kotor atau pencilan liar.

---

### 3.2 Mean Squared Error (MSE / L2 Loss)

#### A. Konsep dan Formula
MSE menghitung **rata-rata kuadrat dari selisih** antara target sebenarnya dengan hasil tebakan:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

#### B. Intuisi Hukuman
MSE memperlakukan kesalahan secara **sangat keras dan berlipat ganda (kurva parabola)**:
* Jika tebakanmu meleset sebesar $2$ poin, hukumannya adalah $2^2 = 4$.
* Jika tebakanmu meleset sebesar $10$ poin, hukumannya meledak menjadi $10^2 = 100$ (**dua puluh lima kali lipat!**).

#### C. Keunggulan Matematis & Kelemahan Praktis
* **Keunggulan**: kurvanya berupa parabola mulus yang turunannya sangat rapi dan mudah dihitung oleh kalkulus, sehingga proses optimasi biasanya berjalan lebih stabil dan cepat menemukan titik terbaik.
* **Kelemahan**: sangat rentan terhadap data pencilan (*outliers*). Jika ada satu data yang meleset jauh, nilai kuadratnya akan mendominasi perhitungan dan memaksa model merusak seluruh tebakan data lainnya hanya demi memuaskan satu data anomali tersebut.

---

### 3.3 Intuisi Visual: Bentuk Kurva Hukuman

```
Nilai Loss ^
           |                     /   (MSE: Kuadrat Melengkung Curam)
           |    \       MSE     /
           |     \             /
           |      \   MAE     /      (MAE: Huruf V Garis Lurus)
           |       \   |     /
           |        \  |    /
           |         \ |   /
           +-----------V------------>
                    Galat (y - y_pred)
```

---

### 3.4 Tabel Komparasi MAE vs MSE
| Parameter | Mean Absolute Error (MAE) | Mean Squared Error (MSE) |
| :--- | :--- | :--- |
| **Operasi Matematika** | Nilai mutlak ($|y - \hat{y}|$) | Kuadrat ($(y - \hat{y})^2$) |
| **Bentuk Kurva** | Sudut tajam huruf V di titik nol | Kurva mangkuk parabola mulus |
| **Sikap pada Galat Besar** | Hukuman bertambah sebanding/linear | Hukuman meledak secara eksponensial |
| **Ketahanan Pencilan** | Sangat tangguh (*robust*) | Sangat rapuh (*sensitive*) |
| **Satuan Angka Hasil** | Sama dengan satuan data asli | Kuadrat dari satuan data asli (butuh akar $\sqrt{\text{MSE}}$) |
| **Kapan Dipilih?** | Banyak anomali kotor pada data | Data relatif bersih dan galat fatal harus dicegah |

---

### 3.5 Sumber Rujukan Topik 3
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 2: “Overview of Supervised Learning (Local Methods in High Dimensions)”.
* **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**, *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2, Springer, Bab 2: “Assessing Model Accuracy”.

---

## 4. Classification Losses: Log Loss / Cross-Entropy (Intuisi)

Pada kasus klasifikasi (misalnya menebak apakah suatu transaksi adalah transaksi normal atau penipuan), targetnya bukan berupa angka kontinu melainkan label kategori diskrit ($0$ atau $1$).

Mengapa kita tidak memakai selisih angka biasa? Karena model klasifikasi modern tidak sekadar menebak label, melainkan mengeluarkan **skor peluang keyakinan (*probability confidence*)** di antara rentang $0,0$ hingga $1,0$.

---

### 4.1 Formula Log Loss (Binary Cross-Entropy)
Untuk satu baris data dengan target $y \in \{0, 1\}$ dan hasil tebakan probabilitas $\hat{y} \in [0, 1]$:

$$L_{\text{log}}(y, \hat{y}) = -\Big[ y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y}) \Big]$$

Jangan panik melihat rumus di atas. Rumus tersebut sebenarnya hanyalah dua kondisi percabangan yang digabung:
* **Jika target asli $y = 1$**: rumus menyusut menjadi $-\ln(\hat{y})$.
* **Jika target asli $y = 0$**: rumus menyusut menjadi $-\ln(1 - \hat{y})$.

---

### 4.2 Intuisi Hukuman Logaritmik: Hukuman untuk Kesombongan yang Salah
Log Loss dirancang dengan prinsip psikologis yang sangat cerdas: **model dihukum sangat berat jika ia sangat yakin pada tebakan yang salah**.

Perhatikan contoh skenario ketika target sebenarnya adalah kelas $1$ (misalnya transaksi memang benar penipuan):
* Jika model menebak peluang $\hat{y} = 0,99$ (model sangat yakin dan tebakannya benar):
  $$-\ln(0,99) \approx 0,01 \quad \text{(Hukuman hampir nol / sangat puas)}$$
* Jika model ragu-ragu dan menebak peluang $\hat{y} = 0,50$:
  $$-\ln(0,50) \approx 0,69 \quad \text{(Hukuman sedang)}$$
* Jika model sombong dan menebak peluang $\hat{y} = 0,01$ (model sangat yakin bahwa ini bukan penipuan, padahal nyatanya penipuan):
  $$-\ln(0,01) \approx 4,60 \quad \text{(Hukuman melonjak tinggi!)}$$
* Jika model menebak $\hat{y} \rightarrow 0$ secara mutlak pada target $1$:
  $$-\ln(0) \rightarrow \infty \quad \text{(Hukuman bernilai tak hingga!)}$$

```
Nilai Loss ^
           | \
           |  \   (Jika Target Nyata y = 1)
           |   \
           |    \   Semakin mendekati tebakan 0,
           |     \  hukuman meledak ke atas!
           |      \
           |       +-------------------->
           0.0     0.5                1.0
             Tebakan Probabilitas (y_pred)
```

Dengan kurva logaritmik ini, model dipaksa untuk tidak hanya menebak kelas yang benar, tetapi juga belajar mengkalibrasi tingkat kepastian probabilitasnya dengan jujur.

---

### 4.3 Sumber Rujukan Topik 4
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 4: “Linear Models for Classification”.
* **Murphy, Kevin P. (2012)**, *Machine Learning: A Probabilistic Perspective*, MIT Press, Bab 9: “Generalized Linear Models and the Exponential Family”.

---

## 5. Mengapa Model Meminimalkan Loss, Bukan Memaksimalkan Akurasi?

Pertanyaan paling mendasar yang sering muncul dari pemula:
> *“Tujuan akhir kita kan ingin model memiliki akurasi $99\%$. Kenapa algoritma latih tidak langsung disuruh memaksimalkan skor akurasi saja? Kenapa harus repot-repot membuat rumus fungsi kerugian?”*

Jawabannya terletak pada **keterbatasan sifat matematika kalkulus**.

---

### 5.1 Masalah Fungsi Tangga pada Akurasi
Akurasi dihitung dari perbandingan jumlah tebakan yang benar dibagi total data:

$$\text{Akurasi} = \frac{\text{Jumlah Benar}}{\text{Total Data}}$$

Perhatikan apa yang terjadi saat model mengubah sedikit parameter bobot dalamnya:
* Misalkan model menebak probabilitas penipuan naik dari $0,21$ menjadi $0,29$.
* Karena ambang batas tebakan biner ada di angka $0,50$, kedua tebakan tersebut tetap dikategorikan sebagai "Bukan Penipuan" (kelas $0$).
* **Hasilnya**: akurasi tidak bergeming sama sekali ($0\%$). Perubahan bobot tidak memicu perubahan nilai akurasi.

Secara grafik, akurasi berbentuk **tangga datar terputus-putus (*step function*)**:

```
Akurasi ^
        |                +----------- (Semua tebakan benar)
        |                |
        |   +------------+            (Fungsi Tangga: Datar!)
        |   |                         Turunan / Kemiringan = 0
        +---+------------------------>
                   Nilai Parameter Bobot (w)
```

Jika kurvanya datar, **nilai kemiringannya (turunan atau gradien) bernilai nol**. 
Jika gradien bernilai nol, algoritma *Gradient Descent* menjadi buta arah. Komputer tidak tahu apakah pergeseran bobot barusan membuat keadaan lebih baik atau lebih buruk.

---

### 5.2 Solusi: Fungsi Kerugian yang Mulus dan Kontinu
Fungsi kerugian (seperti *Log Loss*) tidak menggunakan pembulatan benar/salah secara kaku. Fungsi ini membaca langsung pergeseran angka probabilitas desimal.

* Ketika probabilitas naik dari $0,21$ menjadi $0,29$, nilai *loss* langsung turun (misal dari $1,56$ menjadi $1,23$).
* Karena kurva *loss* **sangat mulus dan bersambung (*smooth and continuous*)**, nilai kemiringan lerengnya (gradien) selalu ada dan tidak pernah nol.
* Lereng kemiringan inilah yang memberi tahu komputer: *"Ayo geser bobotmu ke arah kanan sedikit lagi, karena ke arah sana jurang kerugiannya menurun!"*

```
KOMPARASI KURVA BELAJAR:

AKURASI (Fungsi Tangga):           LOSS FUNCTION (Kurva Mulus):
      |                                  |
      |       +----+                     | \
      |       |                          |  \   <--- Ada lereng kemiringan!
      |  +----+                          |   \       Komputer tahu arah turun.
      +------------>                     +------------>
      BUTA ARAH (Turunan = 0)            MEMILIKI PANDUAN (Turunan != 0)
```

---

### 5.3 Sumber Rujukan Topik 5
* **Nielsen, Michael A. (2015)**, *Neural Networks and Deep Learning*, Determination Press, Bab 1: “Using neural nets to recognize handwritten digits (Why use cross-entropy instead of classification accuracy?)”.
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 4: “Numerical Computation (Gradient-Based Optimization)”.

---

## 6. Rata-Rata Loss pada Seluruh Dataset (*Cost Function*)

Sejauh ini kita membahas fungsi kerugian untuk satu baris data tunggal ($L$). Namun dalam kenyataannya, model kita harus berhadapan dengan ribuan hingga jutaan baris data sekaligus.

### 6.1 Dari Loss Menjadi Cost Function
Ketika kita menghitung rata-rata nilai kerugian dari seluruh dataset, fungsi tersebut resmi disebut sebagai **Fungsi Biaya (*Cost Function*)**, yang biasa dilambangkan dengan huruf $J(\theta)$:

$$J(\theta) = \frac{1}{N} \sum_{i=1}^N L\Big(y_i, f(x_i; \theta)\Big)$$

* $N$ adalah jumlah total baris data latih.
* $\theta$ adalah seluruh kumpulan parameter bobot di dalam model.
* $J(\theta)$ adalah skor kerugian rata-rata yang harus diminimalkan.

Konsep mencari setelan parameter $\theta$ yang menghasilkan nilai rata-rata kerugian terendah pada data historis ini dikenal dalam teori pembelajaran statistik sebagai **Minimisasi Risiko Empiris (*Empirical Risk Minimization / ERM*)**.

---

### 6.2 Tiga Strategi Menghitung Rata-Rata Loss dalam Latihan

Bagaimana cara komputer menghitung rata-rata nilai $J(\theta)$ saat melakukan latihan? Ada tiga pendekatan:

```
1. BATCH GRADIENT DESCENT:
   Baca SELURUH Data (N baris) ---> Hitung Rata-rata Loss ---> Geser Bobot 1 Kali.
   Sifat: Sangat stabil, tetapi sangat lambat dan memakan memori jika data jutaan.

2. STOCHASTIC GRADIENT DESCENT (SGD):
   Baca HANYA 1 Baris Data      ---> Hitung Loss 1 Baris   ---> Geser Bobot Langsung.
   Sifat: Sangat cepat, tetapi lintasannya zig-zag liar dan tidak stabil.

3. MINI-BATCH GRADIENT DESCENT (Standar Industri Modern):
   Baca Sebagian Data (misal: 32, 64, atau 128 baris) ---> Hitung Rata-rata Loss Mini
   ---> Geser Bobot.
   Sifat: Keseimbangan sempurna antara kecepatan perangkat keras (GPU) dan kestabilan.
```

---

### 6.3 Sumber Rujukan Topik 6
* **Vapnik, Vladimir N. (1998)**, *Statistical Learning Theory*, Wiley-Interscience, Bab 1: “The Setting of the Learning Problem (Empirical Risk Minimization Principle)”.
* **Bottou, Léon, Curtis, Frank E., dan Nocedal, Jorge (2018)**, “Optimization Methods for Large-Scale Machine Learning”, *SIAM Review*, 60(2), hlm. 223–311.

---

## 7. Kerangka Rangkuman Alur Kerja Optimasi

Diagram alur sistematis bagaimana fungsi kerugian menggerakkan seluruh proses pembelajaran model:

```
                            [ INISIALISASI MODEL ]
                            (Bobot acak awal: theta)
                                       |
                                       v
                             [ DATA MASUKAN (X) ]
                                       |
                                       v
                            [ LAKUKAN INFERENSI ]
                            (Tebak hasil: y_pred)
                                       |
                                       v
                         [ HITUNG RATA-RATA KERUGIAN ]
                         (J(theta) lewat MAE / MSE / Log-Loss)
                                       |
                                       v
                    +-------------------------------------+
                    | APAKAH NILAI LOSS SUDAH MINIMAL?    |
                    +-------------------------------------+
                                   /       \
                            TIDAK /         \ YA
                                 v           v
                     [ HITUNG GRADIEN ]   [ SELESAI! MODEL TERLATIH ]
                     (Cari turunan dJ/dtheta) (Kunci parameter terbaik)
                                 |
                                 v
                     [ PERBARUI BOBOT MODEL ]
                     (theta_baru = theta - lr * gradien)
                                 |
                                 +---> Ulangi ke Inferensi!
```

---

## 8. Daftar Pustaka Terverifikasi

1. **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**. *Deep Learning*. MIT Press.
2. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
3. **Bishop, Christopher M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.
4. **Murphy, Kevin P. (2012)**. *Machine Learning: A Probabilistic Perspective*. The MIT Press.
5. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
6. **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**. *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2. Springer.
7. **Chollet, François (2021)**. *Deep Learning with Python*, Edisi ke-2. Manning Publications.
8. **Nielsen, Michael A. (2015)**. *Neural Networks and Deep Learning*. Determination Press.
9. **Bottou, Léon, Curtis, Frank E., dan Nocedal, Jorge (2018)**. “Optimization Methods for Large-Scale Machine Learning”. *SIAM Review*, 60(2), hlm. 223–311.
10. **Vapnik, Vladimir N. (1998)**. *Statistical Learning Theory*. Wiley-Interscience.