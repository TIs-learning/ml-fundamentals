# Modul Pembelajaran: ML Problem Formulation

Dokumentasi pembelajaran ini membahas tahap paling krusial dalam siklus rekayasa *Machine Learning* (ML), yaitu **Formulasi Masalah (*Problem Formulation*)**—seni dan sains mentranslasikan kebutuhan bisnis dunia nyata menjadi permasalahan komputasi matematis yang terdefinisi dengan baik (*well-defined mathematical problem*).

---

## Glosarium Istilah Asing Penting
Sebelum mendalami materi, pahami beberapa istilah kunci yang sering digunakan:
* **Fitur (*Features*, $X$)**: variabel masukan atau karakteristik terukur dari suatu data yang digunakan model untuk membuat keputusan.
* **Target / Label ($Y$)**: variabel luaran atau nilai fakta lapangan (*ground truth*) yang ingin diprediksi oleh model.
* **Fakta Lapangan (*Ground Truth*)**: nilai kebenaran mutlak yang terverifikasi di dunia nyata untuk suatu sampel data.
* **Ruang Keadaan (*State Space*)**: rentang seluruh nilai yang mungkin dimiliki oleh suatu variabel atau fungsi.
* **Fungsi Kerugian (*Loss Function*)**: formula matematika yang mengukur seberapa besar selisih atau kesalahan tebakan model dibandingkan nilai kebenaran sebenarnya.
* **Saling Eksklusif (*Mutually Exclusive*)**: kondisi di mana terjadinya satu kejadian atau pemilihan satu kelas secara otomatis menggugurkan kemungkinan terjadinya kelas lainnya.

---

## 1. Pengantar Formulasi Masalah ML

Banyak inisiatif kecerdasan buatan gagal bukan karena kelemahan algoritma, melainkan karena **salah merumuskan masalah (*misalignment of problem formulation*)**.

Secara fundamental, formulasi masalah ML adalah proses menetapkan:
1. **Representasi Masukan ($X$)**: informasi apa yang tersedia saat sistem melakukan inferensi di lingkungan produksi.
2. **Representasi Luaran ($Y$)**: format jawaban apa yang paling optimal untuk mendukung pengambilan keputusan.
3. **Fungsi Objektif**: fungsi matematika yang memandu algoritma dalam membedakan antara prediksi yang baik dan buruk.

```
                           SIKLUS FORMULASI MASALAH ML
                           
  [ Masalah Bisnis Nyata ]
             |
             v
  [ Identifikasi Variabel ] ---> Fitur Masukan (X) vs Target Luaran (Y)
             |
             v
  [ Pemilihan Paradigma ]  ---> Supervised vs Unsupervised vs Ranking
             |
             v
  [ Penetapan Format Y ]   ---> Kontinu (Regresi) vs Diskrit (Klasifikasi)
             |
             v
  [ Metrik Evaluasi & Loss]---> Optimasi Matematis (MSE, Cross-Entropy, NDCG)
```

---

## 2. Regression vs Classification

Dua pilar utama dalam pembelajaran terawasi (*supervised learning*) dibedakan berdasarkan sifat matematis dari variabel targetnya ($Y$).

### 2.1 Pengertian Istilah Khusus
* **Kontinu (*Continuous*)**: nilai yang dapat berupa bilangan riil tak terbatas di dalam rentang tertentu, termasuk nilai desimal dan pecahan (contoh: $170,5$ cm; $\text{Rp}2.500.350,00$).
* **Diskrit (*Discrete*)**: nilai berupa kategori terpisah atau bilangan bulat yang terhitung (contoh: tipe darah A, B, AB, O; status lulus/tidak lulus).

---

### 2.2 Regresi (*Regression*)
Regresi adalah perumusan masalah di mana model memetakan variabel masukan ($X$) ke variabel target luaran ($Y$) yang bernilai **kontinu**:

$$f: \mathbb{R}^d \rightarrow \mathbb{R}$$

* **Tujuan**: mengestimasi besaran skalar atau kuantitas numerik seakurat mungkin.
* **Contoh Kasus**:
  * Memprediksi harga jual rumah berdasarkan luas tanah, lokasi, dan jumlah kamar.
  * Memprediksi konsumsi daya listrik harian sebuah gedung operasional dalam satuan kilowatt-jam (kWh).
  * Memprediksi sisa masa pakai komponen mesin (*Remaining Useful Life*).
* **Fungsi Kerugian (*Loss Function*) Umum**:
  * *Mean Squared Error* (MSE): menghukum galat besar secara eksponensial.
  * *Mean Absolute Error* (MAE): lebih toleran terhadap titik pencilan (*outliers*).

---

### 2.3 Klasifikasi (*Classification*)
Klasifikasi adalah perumusan masalah di mana model memetakan variabel masukan ($X$) ke variabel target luaran ($Y$) yang berupa **kategori diskrit**:

$$f: \mathbb{R}^d \rightarrow \{C_1, C_2, \dots, C_k\}$$

* **Tujuan**: menentukan batas keputusan (*decision boundary*) yang memisahkan ruang fitur ke dalam wilayah kelas-kelas yang berbeda.
* **Contoh Kasus**:
  * Mendeteksi apakah transaksi perbankan merupakan transaksi sah atau penipuan (*fraud*).
  * Menentukan status diagnosis medis pasien (positif terinfeksi atau negatif).
  * Mengkategorikan sentimen ulasan pelanggan (positif, netral, atau negatif).
* **Fungsi Kerugian (*Loss Function*) Umum**:
  * *Binary Cross-Entropy* / *Log-Loss* (untuk dua kelas).
  * *Categorical Cross-Entropy* (untuk banyak kelas).

---

### 2.4 Tabel Perbandingan Regression vs Classification
| Dimensi Evaluasi | Regression | Classification |
| :--- | :--- | :--- |
| **Sifat Variabel Target ($Y$)** | Kuantitatif kontinu ($\mathbb{R}$) | Kualitatif diskrit ($\{0, 1\}$ atau label kelas) |
| **Bentuk Keluaran Model** | Nilai estimasi numerik bebas | Skor probabilitas per kelas ($[0, 1]$) |
| **Representasi Geometris** | Kurva aproksimasi melintasi titik data | Bidang batas pemisah (*decision boundary*) |
| **Metrik Evaluasi Utama** | RMSE, MAE, MAPE, $R^2$ Score | Accuracy, Precision, Recall, F1-Score, ROC-AUC |
| **Kerentanan Pencilan** | Sangat sensitif jika menggunakan loss kuadratik | Relatif lebih stabil terhadap batas ekstrem |

---

### 2.5 Sumber Rujukan Topik 2
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2, Springer, Bab 2: “Overview of Supervised Learning”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 3: “Classification” dan Bab 4: “Training Models”.

---

## 3. Binary vs Multi-Class vs Multi-Label

Dalam ranah klasifikasi, hubungan antara sampel data dan labelnya diklasifikasikan ke dalam tiga skema arsitektural.

### 3.1 Pengertian Istilah Khusus
* **Saling Meniadakan (*Mutual Exclusivity*)**: prinsip bahwa satu objek hanya boleh memiliki satu label identitas pasti dalam satu waktu.
* **Sigmoid**: fungsi aktivasi non-linear yang mengubah sembarang bilangan riil menjadi nilai probabilitas antara $0$ hingga $1$ secara independen.
* **Softmax**: fungsi aktivasi yang mengubah sekumpulan angka skor (*logits*) menjadi distribusi probabilitas yang jumlah total keseluruhannya tepat bernilai $1,0$ ($100\%$).

---

### 3.2 Tiga Paradigma Klasifikasi

```
1. BINARY CLASSIFICATION:
   Input [Foto Hewan] ---> Model ---> [ Kucing ] ATAU [ Bukan Kucing ] (Pilihan Biner)

2. MULTI-CLASS CLASSIFICATION:
   Input [Foto Hewan] ---> Model ---> Tepat Satu dari: {Kucing, Anjing, Kuda, Kelinci}
                                      (Saling Eksklusif, Probabilitas Total = 1,0)

3. MULTI-LABEL CLASSIFICATION:
   Input [Artikel Berita] ---> Model ---> Dapat Memiliki Banyak Label Sekaligus:
                                          [x] Politik
                                          [ ] Olahraga
                                          [x] Ekonomi
                                          (Label Independen)
```

#### A. Binary Classification (Klasifikasi Biner)
* Hanya melibatkan **dua kemungkinan kelas** yang saling berlawanan ($Y \in \{0, 1\}$).
* Lapisan luaran model hanya membutuhkan **satu neuron aktivasi** dengan fungsi **Sigmoid**:
  $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}$$
* *Contoh*: prediksi persetujuan pinjaman bank (Diterima atau Ditolak).

#### B. Multi-Class Classification (Klasifikasi Multikelas)
* Melibatkan **lebih dari dua kelas** ($k > 2$), di mana setiap sampel data **hanya boleh menjadi anggota dari tepat satu kelas saja** (*mutually exclusive*).
* Lapisan luaran model menggunakan **$k$ neuron aktivasi** dengan fungsi **Softmax**:
  $$P(Y = j \mid X) = \frac{e^{z_j}}{\sum_{i=1}^k e^{z_i}}$$
* *Contoh*: identifikasi jenis kendaraan pada gerbang tol (Golongan I, Golongan II, Golongan III, atau Golongan IV).

#### C. Multi-Label Classification (Klasifikasi Multilabel)
* Melibatkan banyak kelas, di mana setiap sampel data **dapat memiliki nol, satu, atau beberapa label sekaligus** secara bersamaan (tidak saling meniadakan).
* Lapisan luaran model menggunakan **$k$ neuron aktivasi independen**, masing-masing menggunakan fungsi **Sigmoid** (bukan Softmax) yang dihitung dengan *Binary Cross-Entropy* tersendiri untuk setiap label.
* *Contoh*:
  * Penandaan topik artikel berita: satu artikel yang sama dapat diberi label "Teknologi", "Bisnis", dan "Kecerdasan Buatan" sekaligus.
  * Diagnosis radiologi dada: satu foto rontgen dapat menunjukkan tanda "Pneumonia" sekaligus "Kardiomegali".

---

### 3.3 Tabel Komparasi Skema Klasifikasi
| Parameter | Binary | Multi-Class | Multi-Label |
| :--- | :--- | :--- | :--- |
| **Jumlah Kemungkinan Kelas** | Tepat dua kelas | Lebih dari dua kelas | Lebih dari dua kelas |
| **Jumlah Label per Sampel** | Tepat satu label | Tepat satu label | Nol, satu, atau banyak label |
| **Sifat Hubungan Kelas** | Komplementer | Saling eksklusif | Independen |
| **Aktivasi Lapisan Akhir** | 1 neuron (Sigmoid) | $k$ neuron (Softmax) | $k$ neuron (Sigmoid mandiri) |
| **Fungsi Kerugian (*Loss*)** | Binary Cross-Entropy | Categorical Cross-Entropy | Multi-Label / Sum of BCE |

---

### 3.4 Sumber Rujukan Topik 3
* **Tsoumakas, Grigorios, dan Katakis, Ioannis (2007)**, “Multi-Label Classification: An Overview”, *International Journal of Data Warehousing and Mining*, 3(3), hlm. 1–13.
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 4: “Linear Models for Classification”.

---

## 4. Clustering vs Density Estimation

Dalam pembelajaran tak terawasi (*unsupervised learning*), kedua paradigma ini mencoba memahami pola sebaran fitur masukan tanpa bantuan variabel target.

### 4.1 Pengertian Istilah Khusus
* **Klaster (*Cluster*)**: kelompok data yang memiliki tingkat kemiripan internal tinggi dan berbeda secara signifikan dengan data di kelompok lain.
* **Fungsi Densitas Probabilitas (*Probability Density Function / PDF*)**: fungsi matematika kontinu yang menggambarkan kemungkinan relatif suatu variabel acak mengambil nilai tertentu di dalam ruang fitur.
* **Non-Parametrik (*Non-Parametric*)**: metode pemodelan yang tidak mengasumsikan bentuk distribusi data tertentu (seperti kurva lonceng normal) secara kaku di awal.

---

### 4.2 Clustering (Pengelompokan Diskrit)
*Clustering* bertujuan **mempartisi atau membagi data ke dalam kelompok-kelompok diskrit** berdasarkan metrik jarak atau kedekatan spasial:
* **Luaran**: setiap titik data diberikan label keanggotaan kelompok (misalnya: Titik A milik Kelompok 1, Titik B milik Kelompok 3).
* **Fokus**: menentukan batas wilayah spasial antarkelompok (*hard boundaries* atau *soft memberships*).
* **Pendekatan Populer**:
  * *K-Means*: meminimalkan varians jarak kuadrat terhadap titik pusat (*centroid*).
  * *DBSCAN*: mengelompokkan data berdasarkan kepadatan titik (*density-connected points*) tanpa menetapkan jumlah klaster di awal.

---

### 4.3 Density Estimation (Estimasi Kepadatan Probabilitas)
*Density Estimation* bertujuan **mengonstruksi fungsi kontinu yang menggambarkan distribusi probabilitas dasar data**, $p(x)$, dari kumpulan titik sampel yang diobservasi:
* **Luaran**: sebuah fungsi matematika kontinu yang dapat menghitung nilai kerapatan probabilitas di titik koordinat mana pun di dalam ruang fitur.
* **Fokus**: memodelkan permukaan distribusi probabilitas secara mulus, bukan sekadar memisahkan kelompok.
* **Pendekatan Populer**:
  * *Parametrik* (misal: *Gaussian Mixture Models* / GMM): mengasumsikan data terbentuk dari kombinasi beberapa kurva lonceng Gaussian.
  * *Non-Parametrik* (misal: *Kernel Density Estimation* / KDE): mengestimasi kerapatan dengan meletakkan kurva halus kecil (*kernel*) di atas setiap titik data tanpa asumsi bentuk kurva awal.

```
CLUSTERING (Partisi Diskrit):
   [ Data ] ---> Dikelompokkan ke: Cluster 1, Cluster 2, atau Cluster 3.
                 (Output: ID kelompok atau vektor bobot keanggotaan)

DENSITY ESTIMATION (Pemodelan Kontinu):
   [ Data ] ---> Membentuk fungsi mulus p(x).
                 (Output: Nilai peluang/kerapatan kontinu pada koordinat x)
```

---

### 4.4 Tabel Perbandingan Clustering vs Density Estimation
| Dimensi | Clustering | Density Estimation |
| :--- | :--- | :--- |
| **Keluaran Utama** | Label partisi diskrit per sampel | Fungsi matematis probabilitas kontinu $p(x)$ |
| **Pertanyaan Inti** | “Titik ini masuk ke kelompok mana?” | “Berapa kerapatan probabilitas kemunculan di titik ini?” |
| **Model Representatif** | K-Means, Agglomerative, DBSCAN | Kernel Density Estimation (KDE), GMM |
| **Interseksi Keduanya** | GMM dapat digunakan untuk *clustering* dengan menetapkan klaster dari distribusi berbobot tertinggi |
| **Manfaat Turunan** | Segmentasi pengguna, kompresi data | Pembangkitan data sintetis, deteksi anomali kuantitatif |

---

### 4.5 Sumber Rujukan Topik 4
* **Silverman, Bernard W. (1986)**, *Density Estimation for Statistics and Data Analysis*, Chapman and Hall/CRC.
* **Murphy, Kevin P. (2012)**, *Machine Learning: A Probabilistic Perspective*, MIT Press, Bab 11: “Mixture Models and the EM Algorithm” dan Bab 25: “Clustering”.

---

## 5. Anomaly / Outlier Detection

### 5.1 Pengertian Istilah Khusus
* **Anomali / Pencilan (*Outlier*)**: titik data yang menyimpang secara signifikan dari karakteristik umum populasi data normal, memunculkan kecurigaan bahwa titik tersebut dihasilkan oleh mekanisme yang berbeda.
* **Ketidakseimbangan Kelas Ekstrem (*Extreme Class Imbalance*)**: situasi di mana kejadian normal mencakup $99,9\%$ dari seluruh data, sedangkan kejadian anomali hanya muncul pada porsi $< 0,1\%$.
* **Deteksi Kebaruan (*Novelty Detection*)**: sistem dilatih hanya menggunakan data normal bersih; data baru yang masuk diuji apakah tergolong variasi baru yang belum pernah dilihat sebelumnya.

---

### 5.2 Karakteristik Perumusan Masalah Anomali
Masalah anomali **tidak efektif** dirumuskan sebagai klasifikasi biner standar (*Supervised Classification*) karena:
1. **Langkanya Data Kasus Negatif**: kasus kecelakaan reaktor nuklir, transaksi pembobolan langka, atau cacat mikro pada chip silikon berjumlah sangat sedikit untuk melatih model klasifikasi standar.
2. **Karakteristik Anomali yang Dinamis**: pelaku penipuan siber selalu mengubah teknik mereka sehingga anomali masa depan tidak memiliki tanda yang identik dengan anomali masa lalu.

---

### 5.3 Taksonomi Pendekatan Anomaly Detection

```
                       STRATEGI ANOMALY DETECTION
                                   |
         +-------------------------+-------------------------+
         |                                                   |
         v                                                   v
   SEMI-SUPERVISED                                     UNSUPERVISED
 (Novelty Detection)                                (Outlier Detection)
- Data latih: 100% normal murni.                  - Data latih: campuran data tanpa label
- Mengunci batas toleransi normal.                 (mengandung derau anomali laten).
- Metode: One-Class SVM, Deep Autoencoders.        - Metode: Isolation Forest, Local Outlier
                                                     Factor (LOF), Elliptic Envelope.
```

1. **Pendekatan Pemisahan Pohon (*Isolation Forest*)**:
   * Bekerja dengan prinsip bahwa titik anomali memiliki sifat “sedikit dan berbeda” sehingga lebih mudah diisolasi melalui pemotongan fitur acak (*random splits*) dibanding titik normal yang padat.
2. **Pendekatan Rekonstruksi (*Autoencoders*)**:
   * Jaringan saraf tiruan dilatih untuk merekonstruksi data normal. Ketika diberikan data anomali, model menghasilkan galat rekonstruksi (*reconstruction error*) yang sangat tinggi karena tidak mampu memampatkan fitur yang belum pernah dipelajarinya.

---

### 5.4 Sumber Rujukan Topik 5
* **Liu, Fei Tony, Ting, Kai Ming, dan Zhou, Zhi-Hua (2008)**, “Isolation Forest”, *Eighth IEEE International Conference on Data Mining (ICDM)*, hlm. 413–422.
* **Chandola, Varun, Banerjee, Arindam, dan Kumar, Vipin (2009)**, “Anomaly Detection: A Survey”, *ACM Computing Surveys*, 41(3), hlm. 1–58.
* **Aggarwal, Charu C. (2017)**, *Outlier Analysis*, Edisi ke-2, Springer.

---

## 6. Ranking & Recommendation Problems

Masalah pemeringkatan (*ranking*) dan rekomendasi (*recommendation*) berfokus pada penyusunan urutan preferensi objek yang paling relevan bagi pengguna berdasarkan riwayat interaksi.

### 6.1 Pengertian Istilah Khusus
* **Umpan Balik Eksplisit (*Explicit Feedback*)**: data preferensi yang diberikan pengguna secara sadar (misal: memberikan rating bintang 1–5 atau menuliskan ulasan teks).
* **Umpan Balik Implisit (*Implicit Feedback*)**: data preferensi yang disimpulkan secara tidak langsung dari jejak aktivitas pengguna (misal: jumlah klik, durasi menonton video, atau riwayat pencarian).
* **Daftar Urutan Relevan (*Ordered List*)**: susunan barang atau dokumen di mana posisi urutan paling atas memiliki nilai relevansi tertinggi.

---

### 6.2 Formulasi Learning to Rank (LTR)
Dalam sistem pencarian informasi (*Information Retrieval*), perumusan masalah dibagi menjadi tiga paradigma:

```
1. POINTWISE APPROACH:
   Setiap dokumen diprediksi skor relevansinya secara mandiri (Regresi/Klasifikasi standar).
   Input (Query, Dokumen A) ---> Prediksi Skor: 4.5
   Input (Query, Dokumen B) ---> Prediksi Skor: 2.1
   Kelemahan: Mengabaikan urutan relatif antar-dokumen.

2. PAIRWISE APPROACH:
   Model membandingkan dua dokumen sekaligus untuk menentukan mana yang lebih relevan.
   Input: Query + (Dokumen A, Dokumen B) ---> Prediksi: Dokumen A > Dokumen B.
   Metode: RankNet, LambdaMART.

3. LISTWISE APPROACH:
   Model mengevaluasi dan mengoptimalkan seluruh daftar dokumen sekaligus terhadap fungsi metrik.
   Input: Query + Seluruh Dokumen ---> Optimasi langsung pada posisi urutan daftar.
   Metode: ListNet, LambdaRank.
```

---

### 6.3 Arsitektur Sistem Rekomendasi Industri (Two-Stage Architecture)
Pada platform berskala besar dengan jutaan katalog barang (seperti YouTube, Netflix, atau Tokopedia), menghasilkan rekomendasi langsung menggunakan model rumit membutuhkan waktu komputasi yang terlalu tinggi. Oleh karena itu, masalah ini dipecah menjadi dua tahap:

```
+-----------------------------------------------------------------------------------+
|               ARSITEKTUR DUA TAHAP SISTEM REKOMENDASI (TWO-STAGE)                 |
+-----------------------------------------------------------------------------------+
   [ Jutaan Katalog ]
           |
           v
   +-----------------------+   Metode: Collaborative Filtering, Matrix Factorization,
   | TAHAP 1: RETRIEVAL    |           Two-Tower Vector Embeddings (Approximate NN).
   | (Candidate Generation)|   Tujuan: Menyaring jutaan barang menjadi ~100-500 kandidat
   +-----------------------+           dengan latensi < 20 milidetik.
           |
           v
   +-----------------------+   Metode: Deep Learning Ranking Models, GBDT, Multi-task
   | TAHAP 2: RANKING      |           Learning (memprediksi probabilitas klik & tonton).
   | (Scoring & Re-ranking)|   Tujuan: Mengurutkan ratusan kandidat menjadi 10 rekomendasi
   +-----------------------+           terbaik secara akurat.
           |
           v
   [ 10 Rekomendasi Teratas untuk Pengguna ]
```

---

### 6.4 Metrik Evaluasi Khusus Ranking
Karena akurasi standar tidak memperhitungkan posisi urutan, evaluasi sistem perankingan menggunakan metrik berbobot posisi:
* **Mean Reciprocal Rank (MRR)**: mengukur posisi urutan item pertama yang relevan dengan kebutuhan pengguna.
* **Normalized Discounted Cumulative Gain (NDCG@K)**: menghitung total skor relevansi seluruh item dalam daftar rekomendasi seraya memberikan penalti bobot logaritmik jika item yang relevan berada di posisi bawah.

---

### 6.5 Sumber Rujukan Topik 6
* **Liu, Tie-Yan (2009)**, “Learning to Rank for Information Retrieval”, *Foundations and Trends in Information Retrieval*, 3(3), hlm. 225–331.
* **Covington, Paul, Adams, Jay, dan Sargin, Emre (2016)**, “Deep Neural Networks for YouTube Recommendations”, *Proceedings of the 10th ACM Conference on Recommender Systems (RecSys)*, hlm. 191–198.
* **Manning, Christopher D., Raghavan, Prabhakar, dan Schütze, Hinrich (2008)**, *Introduction to Information Retrieval*, Cambridge University Press, Bab 6: “Scoring, term weighting and the vector space model”.

---

## 7. Kerangka Kerja Keputusan Formulasi Masalah

Gunakan diagram pohon keputusan berikut untuk menentukan perumusan masalah ML yang tepat sesuai kebutuhan sistem:

```
                       [ MULAI DARI PERTANYAAN BISNIS ]
                                      |
         +----------------------------+----------------------------+
         |                                                         |
Apakah ada target luaran pasti?                         Tidak ada target luaran pasti
         |                                                         |
         v                                                         v
   [ SUPERVISED ]                                           [ UNSUPERVISED ]
         |                                                         |
         +-------------------------+                 +-------------+-------------+
         |                         |                 |                           |
Apakah target berupa kuantitas?    |          Ingin mengelompokkan       Ingin memodelkan
         |                         |          sampel secara terpisah?    bentuk sebaran data?
    +----+----+                    |                 |                           |
    |         |                    |                 v                           v
  [ YA ]   [ TIDAK ]               |            CLUSTERING               DENSITY ESTIMATION
    |         |                    |            (K-Means, DBSCAN)            (GMM, KDE)
    v         v                    |
 REGRESSION   |                    |
(MSE/MAE)     |                    |
              |                    +------------------------------------+
              v                                                         |
   Apakah memilih kategori atau urutan?                                 |
              |                                                         |
       +------+------+                                                  |
       |             |                                                  |
   [ KATEGORI ]  [ URUTAN ]                                             |
       |             |                                                  v
  KLASIFIKASI     RANKING                                     Apakah tujuannya mencari
       |        (LTR, NDCG)                                   kejadian super langka?
       |                                                                |
       +--------------------+--------------------+                      v
       |                    |                    |              ANOMALY DETECTION
   2 Pilihan?        >2 Pilihan Eksklusif?  Banyak Opsi Bebas? (Isolation Forest)
       |                    |                    |
       v                    v                    v
     BINARY            MULTI-CLASS          MULTI-LABEL
   (Sigmoid)            (Softmax)          (Multi-Sigmoid)
```

---

## 8. Daftar Pustaka Terverifikasi

1. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
2. **Bishop, Christopher M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.
3. **Murphy, Kevin P. (2012)**. *Machine Learning: A Probabilistic Perspective*. The MIT Press.
4. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
5. **Manning, Christopher D., Raghavan, Prabhakar, dan Schütze, Hinrich (2008)**. *Introduction to Information Retrieval*. Cambridge University Press.
6. **Liu, Tie-Yan (2009)**. “Learning to Rank for Information Retrieval”. *Foundations and Trends in Information Retrieval*, 3(3), hlm. 225–331.
7. **Aggarwal, Charu C. (2017)**. *Outlier Analysis*, Edisi ke-2. Springer International Publishing.
8. **Liu, Fei Tony, Ting, Kai Ming, dan Zhou, Zhi-Hua (2008)**. “Isolation Forest”. *Eighth IEEE International Conference on Data Mining*, hlm. 413–422.
9. **Covington, Paul, Adams, Jay, dan Sargin, Emre (2016)**. “Deep Neural Networks for YouTube Recommendations”. *ACM Conference on Recommender Systems (RecSys)*, hlm. 191–198.
10. **Tsoumakas, Grigorios, dan Katakis, Ioannis (2007)**. “Multi-Label Classification: An Overview”. *International Journal of Data Warehousing and Mining*, 3(3), hlm. 1–13.