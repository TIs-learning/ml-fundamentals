# Modul Pembelajaran: Types of Machine Learning

Dokumen pembelajaran ini menyajikan taksonomi komprehensif sistem Machine Learning (ML) berdasarkan tiga dimensi arsitektural utama:
1. **Berdasarkan derajat supervisi manusia (*amount of human supervision*)**: Supervised, Unsupervised, Semi-Supervised, Self-Supervised, dan Reinforcement Learning.
2. **Berdasarkan kemampuan beradaptasi secara bertahap (*incremental learning on the fly*)**: Batch Learning (Offline) vs Online Learning (Incremental).
3. **Berdasarkan mekanisme generalisasi (*generalization approach*)**: Instance-Based Learning (Lazy) vs Model-Based Learning (Eager).

---

## 1. Taksonomi Sistem Machine Learning

Sistem Machine Learning dapat dikelompokkan ke dalam tiga dimensi utama yang saling melengkapi (*orthogonal*):


```

```
                              +---------------------------------------+
                              |    TAKSONOMI SISTEM MACHINE LEARNING  |
                              +---------------------------------------+
                                                 |
    +--------------------------------------------+--------------------------------------------+
    |                                            |                                            |
    v                                            v                                            v

```

[ DERAJAT SUPERVISI ]                     [ CARA SISTEM BERADAPTASI ]                  [ MEKANISME GENERALISASI ]

* Supervised Learning                     - Batch Learning (Offline)                   - Instance-Based Learning (Lazy)
* Unsupervised Learning                   - Online Learning (Incremental)              - Model-Based Learning (Eager)
* Semi-Supervised Learning
* Self-Supervised Learning
* Reinforcement Learning

```

Ketiga dimensi ini dapat dipadukan dalam satu solusi:
* Algoritma *k-Nearest Neighbors* (k-NN) merupakan model **Supervised**, umumnya dijalankan secara **Batch**, dan berkarakteristik **Instance-Based**.
* Algoritma *Linear Regression* dengan *Stochastic Gradient Descent* (SGD) merupakan model **Supervised**, dapat beroperasi secara **Online**, dan berkarakteristik **Model-Based**.

---

## 2. Klasifikasi Berdasarkan Derajat Supervisi Manusia

### 2.1 Supervised Learning

#### A. Definisi dan Mekanisme
Pada *Supervised Learning*, algoritma dilatih menggunakan himpunan data yang memuat pasangan fitur masukan ($X$) dan label luaran target ($Y$). Model mempelajari fungsi pemetaan matematis:

$$Y = f(X) + \epsilon$$

Tujuannya adalah mengestimasi parameter fungsi sedemikian rupa sehingga ketika diberikan data baru ($X_{\text{baru}}$), model dapat memprediksi nilai $\hat{Y}$ dengan tingkat presisi tinggi.

#### B. Dua Cabang Utama:
1. **Regresi (*Regression*)**:
   * Nilai target ($Y$) bersifat kontinu atau numerik riil ($\mathbb{R}$).
   * *Contoh*: prediksi harga properti rumah, peramalan temperatur cuaca harian, dan estimasi gaji lulusan perguruan tinggi.
   * *Algoritma*: Linear Regression, Ridge/Lasso, Support Vector Regression (SVR), Random Forest Regressor, dan Gradient Boosting (XGBoost/LightGBM).
2. **Klasifikasi (*Classification*)**:
   * Nilai target ($Y$) bersifat diskrit atau kategoris (terbagi ke dalam kelas-kelas).
   * Terbagi menjadi biner (dua kelas, seperti deteksi surel spam vs bukan spam) dan multikelas (lebih dari dua kelas, seperti klasifikasi digit angka 0–9 pada MNIST).
   * *Algoritma*: Logistic Regression, Support Vector Machines (SVM), Decision Tree, Random Forest Classifier, dan Naïve Bayes.

---

### 2.2 Unsupervised Learning

#### A. Definisi dan Mekanisme
Dataset pada *Unsupervised Learning* hanya memuat fitur masukan ($X$) tanpa label atau variabel target ($Y$). Algoritma bertugas mengekstraksi struktur internal, pola sebaran tersembunyi (*latent patterns*), serta korelasi intrinsik secara mandiri.

#### B. Empat Tugas Pokok:
1. **Pengelompokan (*Clustering*)**:
   * Mempartisi data ke dalam klaster-klaster berdasarkan ukuran kedekatan jarak atau densitas fitur.
   * *Contoh*: segmentasi profil pelanggan pada platform niaga elektronik (*e-commerce*).
   * *Algoritma*: K-Means, Hierarchical Clustering, dan DBSCAN.
2. **Reduksi Dimensi (*Dimensionality Reduction*)**:
   * Memadatkan jumlah fitur masukan berdimensi tinggi tanpa membuang sebagian besar varians pentingnya, sekaligus mengatasi *curse of dimensionality* dan memfasilitasi visualisasi 2D/3D.
   * *Contoh*: kompresi data citra resolusi tinggi atau pereduksian ratusan indikator keuangan.
   * *Algoritma*: Principal Component Analysis (PCA), t-SNE, dan UMAP.
3. **Deteksi Anomali (*Anomaly Detection*)**:
   * Mengidentifikasi data asing (*outliers*) yang perilakunya menyimpang drastis dari mayoritas populasi normal.
   * *Contoh*: identifikasi transaksi kartu kredit mencurigakan dan deteksi cacat perakitan komponen pabrik.
   * *Algoritma*: Isolation Forest, One-Class SVM, dan Local Outlier Factor (LOF).
4. **Pembelajaran Aturan Asosiasi (*Association Rule Learning*)**:
   * Menemukan relasi ketergantungan antar-item dalam basis data transaksi skala besar (*Market Basket Analysis*).
   * *Contoh klasik*: analisis pola belanja yang menemukan korelasi pembelian popok bayi bersamaan dengan produk minuman kaleng.
   * *Algoritma*: Apriori, Eclat, dan FP-Growth.

---

### 2.3 Semi-Supervised Learning

#### A. Konsep Dasar
Pelabelan data secara manual memerlukan biaya tinggi, waktu lama, dan keahlian manusia khusus (misalnya dokter spesialis untuk menandai citra tomografi terkomputasi). Sebaliknya, data mentah tanpa label sangat mudah dikumpulkan. *Semi-Supervised Learning* memadukan:
* Sebagian kecil data berlabel ($D_L$).
* Sebagian besar data tanpa label ($D_U$).

#### B. Alur Kerja Pseudo-Labeling:
1. Model awal dilatih menggunakan subset data berlabel ($D_L$).
2. Model tersebut digunakan untuk memprediksi probabilitas label pada subset data tanpa label ($D_U$).
3. Sampel data tanpa label yang memperoleh skor keyakinan (*confidence threshold*) sangat tinggi diberi label semu (*pseudo-labels*).
4. Sampel berlabel semu digabungkan ke himpunan data latih untuk melatih ulang model secara iteratif.

*Contoh Implementasi*: Google Photos secara otomatis mengelompokkan wajah orang yang sama pada ribuan foto tanpa label via *clustering*. Pengguna hanya perlu memberikan nama pada satu foto, lalu sistem mengasosiasikan identitas tersebut ke seluruh foto pada klaster yang sama.

---

### 2.4 Self-Supervised Learning (Pengantar Konseptual)

#### A. Paradigma Supervisi Mandiri
*Self-Supervised Learning* (SSL) memanfaatkan data mentah tanpa anotasi manual untuk menghasilkan label supervisinya sendiri dari struktur internal data (*supervision from the data itself*).

Yann LeCun menjelaskan kedudukan SSL melalui analogi kue (*The Cake Analogy*):
* **Reinforcement Learning** diibaratkan ceri di atas kue (hanya menerima sedikit umpan balik skalar imbalan).
* **Supervised Learning** diibaratkan lapisan gula di atas kue (terbatas pada ribuan bit informasi label manusia).
* **Self-Supervised Learning** diibaratkan adonan utama kue tersebut karena menyerap informasi masif secara langsung dari hubungan antarbagian data.

#### B. Mekanisme Pretext Tasks:
* **Masked Language Modeling (NLP)**: menyembunyikan sebagian kata dalam kalimat untuk ditebak oleh model (fondasi pelatihan arsitektur BERT).
* **Next-Token Prediction (NLP)**: melatih model memprediksi kata berikutnya secara sekuensial (fondasi arsitektur Generative Pretrained Transformer / GPT).
* **Contrastive Learning (Computer Vision)**: merekatkan representasi dua augmentasi dari citra yang sama dan merenggangkannya dari citra lain (SimCLR, MoCo).

---

### 2.5 Reinforcement Learning (Tingkat Tinggi)

#### A. Paradigma Interaksi Agen dan Lingkungan
Reinforcement Learning (RL) beroperasi tanpa pasangan data input-output statis. Sebuah **Agen (*Agent*)** berinteraksi secara mandiri di dalam sebuah **Lingkungan (*Environment*)** melalui mekanisme aksi dan reaksi (*trial and error*).


```

```
                  +-------------------+
                  |    ENVIRONMENT    |
                  +-------------------+
                    |               ^
    State / Keadaan |               | Tindakan / Action (A_t)
         (S_t)      |               |
    Reward / Imbalan|               |
         (R_t)      v               |
                  +-------------------+
                  |       AGENT       |
                  +-------------------+

```

```

#### B. Komponen Inti:
1. **Agen (*Agent*)**: entitas pengambil keputusan (algoritma/model).
2. **Lingkungan (*Environment*)**: ruang tempat agen beroperasi (papan permainan catur, simulator jalan raya, atau pendingin server).
3. **Status (*State*, $S$)**: representasi kondisi lingkungan saat ini yang dapat diamati oleh agen.
4. **Tindakan (*Action*, $A$)**: pilihan manuver atau langkah yang dieksekusi agen.
5. **Imbalan (*Reward*, $R$)**: sinyal numerik skalar berupa hadiah (+ skor) atau hukuman (- skor) atas tindakan yang diambil.
6. **Kebijakan (*Policy*, $\pi$)**: strategi yang memetakan status lingkungan ke tindakan optimal: $\pi(a|s)$.

*Contoh Implementasi*: sistem DeepMind AlphaGo yang mengalahkan juara dunia permainan Go (2016), algoritma navigasi kendaraan otonom, dan optimasi efisiensi pendingin pusat data Google.

---

### 2.6 Sumber dan Rujukan Topik 2
* **CampusX (Nitish Singh)**, “Types of Machine Learning for Beginners | Types of Machine learning in Hindi | Types of ML in Depth”, YouTube Video ID: `81ymPYEtFOw`.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “The Machine Learning Landscape”.
* **Sutton, Richard S., dan Barto, Andrew G. (2018)**, *Reinforcement Learning: An Introduction*, Edisi ke-2, The MIT Press.
* **LeCun, Yann, dan Misra, Ishan (2021)**, “Self-supervised learning: The dark matter of intelligence”, Meta AI Research.

---

## 3. Klasifikasi Berdasarkan Cara Model Belajar dan Beradaptasi

### 3.1 Batch Learning (Offline Learning)

#### A. Mekanisme Operasional
Pada *Batch Learning*, model dilatih menggunakan seluruh dataset yang tersedia secara serempak (*in one batch*). Karena proses komputasi membutuhkan waktu dan sumber daya besar, pelatihan dijalankan secara *offline* di lingkungan pengembangan.

Setelah diunggah ke peladen produksi (*production server*), model berada dalam kondisi beku (*frozen model*). Model hanya melayani permintaan inferensi tanpa memperbarui parameter bobotnya saat menerima masukan data baru.


```

+---------------------+     +------------------+     +-----------------------+
| Seluruh Data Latih  | --> | Pelatihan Offline| --> | Model Statis Dideploy |
| (Batch Lengkap)     |     | (Klaster GPU)    |     | (Hanya Melayani Inferensi)
+---------------------+     +------------------+     +-----------------------+

```

#### B. Keterbatasan:
* **Keterlambatan Penyesuaian Data (*Staleness*)**: model tidak peka terhadap tren baru hingga jadwal pelatihan ulang berkala berikutnya dijalankan.
* **Beban Komputasi Tinggi**: proses pelatihan ulang mengharuskan penggabungan seluruh data historis lama dengan data baru, memicu peningkatan biaya operasional infrastruktur.
* **Keterbatasan Sistem Terisolasi**: tidak dapat diterapkan secara fleksibel pada perangkat dengan koneksi terbatas (*isolated edge devices*).

---

### 3.2 Online Learning (Incremental Learning)

#### A. Mekanisme Operasional
Pada *Online Learning* (disebut pula *Incremental/Streaming Learning*), model dilatih secara berkesinambungan dengan memasukkan aliran data baru secara sekuensial, baik per satu sampel (*single instance*) maupun per kelompok kecil (*mini-batches*).


```

Aliran Data Masuk (Streaming) ---> [ Mini-batch / Instance ] ---> [ Model di Server ]
|
v
Pembaruan Bobot Otomatis
(Inferensi & Belajar Langsung)

```

Data yang sudah dipelajari dapat langsung dibuang dari media penyimpanan, sehingga sangat menghemat kapasitas memori.

#### B. Parameter Kunci: Laju Pembelajaran (*Learning Rate*)
* **Learning Rate Tinggi**: model beradaptasi cepat terhadap pola data terkini, tetapi rentan mengalami kondisi lupa katastropik (*catastrophic forgetting*) terhadap informasi lama.
* **Learning Rate Rendah**: model lebih stabil dan tahan terhadap derau, tetapi lambat merespons pergeseran tren baru.

#### C. Penanganan Pergeseran Konsep (*Concept Drift*)
*Concept Drift* adalah fenomena pergeseran distribusi statistik antara data masukan dan target seiring berjalannya waktu (seperti perubahan preferensi transaksi konsumen atau fluktuasi nilai pasar modal). *Online Learning* dirancang secara adaptif untuk memperbarui bobot parameter mengikuti perubahan tersebut.

---

### 3.3 Out-of-Core Learning

Teknik pelatihan dataset berskala masif yang kapasitasnya melampaui batas memori utama (RAM) komputer fisik (misalnya data berukuran 250 GB pada mesin berkapasitas RAM 16 GB).

Tahapan eksekusi:
1. Dataset di media penyimpanan (*disk*) dibagi menjadi segmen-segmen kecil (*chunks*).
2. Segmen dimuat ke memori secara bergantian, kemudian parameter model diperbarui menggunakan algoritma inkremental (seperti fungsi `partial_fit()` pada Scikit-Learn).
3. Memori dibersihkan dari segmen tersebut sebelum memuat segmen berikutnya hingga seluruh data disk selesai diproses.

---

### 3.4 Risiko Keracunan Data (*Data Poisoning*) dan Langkah Mitigasi

Mengizinkan model memperbarui bobotnya secara otomatis di peladen produksi membuka celah kerentanan operasional:
* **Risiko Serangan Siber**: jika data aliran masukan terkontaminasi oleh anomali atau data manipulasi sengaja (*adversarial attacks*), kinerja model dapat terdegradasi seketika.
* **Langkah Mitigasi**:
  1. Memasang saringan deteksi anomali pada gerbang masukan sebelum data diserap oleh model.
  2. Menyediakan sistem pemantauan metrik secara seketika (*real-time alerting*) yang otomatis menonaktifkan pembelajaran dinamis saat terjadi penurunan akurasi drastis.
  3. Menyimpan titik pemulihan bobot (*checkpoints*) secara teratur agar model dapat dikembalikan (*rollback*) ke versi stabil terdahulu.

---

### 3.5 Tabel Komparasi Batch vs Online Learning

| Parameter Evaluasi | Batch Learning (Offline) | Online Learning (Incremental) |
| :--- | :--- | :--- |
| **Aliran Data Latih** | Seluruh dataset secara serempak | Bertahap per instans atau *mini-batch* |
| **Pembaruan Bobot** | Terjadwal di luar lingkungan produksi | Waktu nyata (*on the fly*) di peladen produksi |
| **Kebutuhan Memori RAM** | Sangat tinggi (sebanding skala dataset) | Rendah (hanya memuat data aktif) |
| **Adaptabilitas *Concept Drift*** | Lambat (menunggu siklus pelatihan ulang) | Sangat cepat dan dinamis |
| **Arsip Data Historis** | Seluruh data mentah wajib disimpan | Data lama dapat dibuang setelah dipelajari |
| **Stabilitas Sistem** | Stabil dan mudah diaudit | Memerlukan arsitektur pengawasan ketat |
| **Dukungan Pustaka Python** | Standar Scikit-Learn (`fit`) | Scikit-Learn (`partial_fit`), River, Vowpal Wabbit |

---

### 3.6 Sumber dan Rujukan Topik 3
* **CampusX (Nitish Singh)**, “Batch Machine Learning | Offline Vs Online Learning | Machine Learning Types”, YouTube Video ID: `nPrhFxEuTYU`.
* **CampusX (Nitish Singh)**, “Online Machine Learning | Online Learning | Online Vs Offline Machine Learning”, YouTube Video ID: `3oOipgCbLIk`.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “Batch and Online Learning”.
* **Montiel, Jacob, dkk. (2021)**, “River: machine learning for streaming data in Python”, *Journal of Machine Learning Research (JMLR)*, 22(110), hlm. 1–8.

---

## 4. Klasifikasi Berdasarkan Mekanisme Generalisasi

Berdasarkan cara sistem melakukan generalisasi terhadap titik data kueri baru (*unseen instances*), pendekatan pembelajaran mesin terbagi menjadi **Instance-Based Learning** dan **Model-Based Learning**.


```

```
                              +---------------------------------------+
                              |         MEKANISME GENERALISASI        |
                              +---------------------------------------+
                                   /                                 \
                                  /                                   \
                                 v                                     v
             +-----------------------------------+     +-----------------------------------+
             |    INSTANCE-BASED LEARNING (LAZY) |     |     MODEL-BASED LEARNING (EAGER)  |
             |   Menghafal Titik Data Mentah     |     |    Mengekstraksi Parameter Fungsi |
             +-----------------------------------+     +-----------------------------------+

```

```

---

### 4.1 Instance-Based Learning (Lazy Learning)

#### A. Prinsip Operasional
Pada *Instance-Based Learning*:
* Algoritma mempelajari pola dengan cara **menghafal seluruh data pelatihan mentah** ke dalam memori (*memorization*).
* Saat fase pelatihan, sistem tidak menjalankan komputasi optimasi fungsi matematika apa pun sehingga waktu komputasi pelatihan bernilai instan ($O(1)$). Karakteristik ini membuatnya dijuluki sebagai **Lazy Learning**.
* Generalisasi baru dihitung saat ada titik data kueri baru yang masuk. Sistem mengukur jarak kesamaan (*similarity/distance metric*, seperti Euclidean atau Manhattan) antara data baru tersebut dengan data historis yang tersimpan.

#### B. Contoh Representatif: k-Nearest Neighbors (k-NN)
Jika terdapat data kueri baru $(x_1, x_2)$:
1. Algoritma menghitung jarak terhadap seluruh observasi pada himpunan data.
2. Memilih sejumlah $k$ observasi terdekat.
3. Menetapkan label luaran berdasarkan suara terbanyak (*majority vote*) untuk klasifikasi, atau menghitung nilai rata-rata tetangga terdekat untuk regresi.

---

### 4.2 Model-Based Learning (Eager Learning)

#### A. Prinsip Operasional
Pada *Model-Based Learning*:
* Algoritma secara aktif menggunakan data pelatihan untuk **mengestimasi parameter matematis** dari suatu fungsi hipotesis sebelum data kueri masuk. Karakteristik ini membuatnya dijuluki sebagai **Eager Learning**.
* Melalui algoritma optimasi (*Ordinary Least Squares* atau *Gradient Descent*), model membangun batas keputusan (*decision boundary*) atau fungsi estimasi:

$$f(X) = W^T X + b$$

* Setelah parameter bobot ($W$) dan bias ($b$) optimal terbentuk, **seluruh data mentah pelatihan dapat dilepas dari memori**. Saat inferensi berlangsung, data kueri cukup dimasukkan ke dalam persamaan matematis tersebut.

---

### 4.3 Tabel Komparasi Instance-Based vs Model-Based Learning

| Dimensi Evaluasi | Instance-Based Learning (Lazy) | Model-Based Learning (Eager) |
| :--- | :--- | :--- |
| **Prinsip Dasar** | Menghafal seluruh sampel data (*memorizing*) | Mengekstraksi parameter fungsi (*generalizing*) |
| **Durasi Komputasi Latih** | Instan ($O(1)$) | Intensif dan memakan daya komputasi |
| **Durasi Komputasi Prediksi** | Lambat ($O(N)$ terhadap total observasi) | Sangat cepat ($O(d)$ terhadap jumlah fitur) |
| **Kebutuhan Memori Model** | Besar (wajib menyimpan seluruh data latih) | Ringkas (hanya menyimpan berkas bobot/parameter) |
| **Ketergantungan Data Latih**| Data latih wajib ada saat inferensi berjalan | Data latih dapat dibuang setelah pelatihan tuntas |
| **Contoh Algoritma** | k-Nearest Neighbors (k-NN), Kernel Density | Regresi Linear, SVM, Decision Tree, Neural Networks |

---

### 4.4 Sumber dan Rujukan Topik 4
* **CampusX (Nitish Singh)**, “Instance-Based Vs Model-Based Learning | Types of Machine Learning”, YouTube Video ID: `ntAOq1ioTKo`.
* **Aha, David W., Kibler, Dennis, dan Albert, Marc K. (1991)**, “Instance-based learning algorithms”, *Machine Learning*, 6(1), hlm. 37–66.
* **Mitchell, Tom M. (1997)**, *Machine Learning*, McGraw-Hill, Bab 8: “Instance-Based Learning”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “Instance-Based Versus Model-Based Learning”.

---

## 5. Kerangka Kerja Keputusan Pemilihan Arsitektur ML

Diagram alur logis pemilihan kombinasi paradigma sistem Machine Learning:


```

```
                        [ Analisis Masalah Rekayasa ]
                                       |
            +--------------------------+--------------------------+
            |                                                     |
Apakah data memiliki label target?                 Bagaimana dinamika aliran data?
            |                                                     |
   +--------+--------+                                   +--------+--------+
   |                 |                                   |                 |
 [ YA ]           [ TIDAK ]                          [ STATIS ]       [ DINAMIS ]
   |                 |                               (Berkala)        (Streaming)
   v                 v                                   |                 |

```

SUPERVISED       UNSUPERVISED                              v                 v
(Regresi atau    (Clustering,                            BATCH            ONLINE
Klasifikasi)     Reduksi Dimensi)                     LEARNING          LEARNING
|                 |                                   |                 |
+--------+--------+                                   +--------+--------+
|                                                     |
+--------------------------+--------------------------+
|
Bagaimana toleransi latensi inferensi?
|
+-----------------+-----------------+
|                                   |
[ SANGAT KETAT ]                    [ FLEKSIBEL ]
(Inference < 10ms)                 (Dataset Kecil/Lokal)
|                                   |
v                                   v
MODEL-BASED                       INSTANCE-BASED
(Linear/Trees/NN)                      (k-NN/RBF)

```

---

## 6. Daftar Referensi dan Rujukan Terverifikasi

1. **CampusX (Nitish Singh)**:
   * “Types of Machine Learning for Beginners | Types of Machine learning in Hindi | Types of ML in Depth”, URL: `https://www.youtube.com/watch?v=81ymPYEtFOw`.
   * “Batch Machine Learning | Offline Vs Online Learning | Machine Learning Types”, URL: `https://www.youtube.com/watch?v=nPrhFxEuTYU`.
   * “Online Machine Learning | Online Learning | Online Vs Offline Machine Learning”, URL: `https://www.youtube.com/watch?v=3oOipgCbLIk`.
   * “Instance-Based Vs Model-Based Learning | Types of Machine Learning”, URL: `https://www.youtube.com/watch?v=ntAOq1ioTKo`.
2. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
3. **Sutton, Richard S., dan Barto, Andrew G. (2018)**. *Reinforcement Learning: An Introduction*, Edisi ke-2. The MIT Press.
4. **LeCun, Yann, dan Misra, Ishan (2021)**. “Self-supervised learning: The dark matter of intelligence”. Meta AI Research.
5. **Montiel, Jacob, dkk. (2021)**. “River: machine learning for streaming data in Python”. *Journal of Machine Learning Research (JMLR)*, 22(110), hlm. 1–8.
6. **Mitchell, Tom M. (1997)**. *Machine Learning*. McGraw-Hill Education.
7. **Aha, David W., Kibler, Dennis, dan Albert, Marc K. (1991)**. “Instance-based learning algorithms”. *Machine Learning*, 6(1), hlm. 37–66.