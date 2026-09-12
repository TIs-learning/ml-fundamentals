# Modul Pembelajaran: Evaluation Metrics (Conceptual)

Dokumentasi pembelajaran ini membahas instrumen penilaian performa model *Machine Learning* (ML): **Metrik Evaluasi (*Evaluation Metrics*)**. Di sini kita akan membedah bagaimana cara manusia mengukur kinerja model secara objektif pada kasus regresi dan klasifikasi, mengapa akurasi sering kali menjadi metrik yang menyesatkan, serta bagaimana cara memilih metrik yang tepat sesuai risiko bisnis di dunia nyata.

---

## Glosarium Istilah Penting
Pahami istilah kunci berikut agar pembahasan terasa membumi:
* **Metrik Evaluasi (*Evaluation Metric*)**: angka tolok ukur yang digunakan manusia untuk menilai seberapa cerdas, bermanfaat, dan akurat hasil tebakan model.
* **Fakta Lapangan (*Ground Truth*, $y$)**: nilai atau status nyata yang benar-benar terjadi di lapangan.
* **Hasil Tebakan (*Predictions*, $\hat{y}$)**: angka atau label yang dikeluarkan oleh model saat menguji data.
* **Model Baseline Polos (*Naive Baseline*)**: model tebakan paling bodoh tanpa algoritma pintar, misalnya selalu menebak nilai rata-rata pada regresi atau selalu menebak kelas mayoritas pada klasifikasi.
* **Data Timpang (*Imbalanced Data*)**: kondisi sebaran dataset di mana salah satu kelompok target berjumlah sangat dominan (misalnya $99\%$), sedangkan kelompok lainnya sangat langka (misalnya $1\%$).
* **Rata-Rata Harmonik (*Harmonic Mean*)**: jenis perhitungan rata-rata khusus yang memberikan bobot hukuman sangat berat jika salah satu komponen bernilai mendekati nol.

---

## 1. Pengantar: Evaluasi Model dalam Rekayasa ML

Banyak pemula mengira pekerjaan merekayasa model selesai saat nilai fungsi kerugian (*loss*) mengecil. Padahal, **komputer meminimalkan loss untuk belajar, sedangkan manusia membaca metrik untuk mengambil keputusan bisnis.**

```
+-----------------------------------------------------------------------------+
|                     LOSS FUNCTION VS EVALUATION METRIC                      |
+-----------------------------------------------------------------------------+

  [ PROSES BELAJAR / LATIHAN ]                 [ PENILAIAN PERFORMA ]
  Konsumen: Algoritma (Komputer)              Konsumen: Manusia (Engineer & Bisnis)
  Fokus   : Menggeser parameter bobot         Fokus   : Apakah model layak dirilis?
  Sifat   : Wajib mulus & bisa diturunkan     Sifat   : Intuitif & mudah dimengerti
  Contoh  : MSE, Log-Loss                     Contoh  : MAE, F1-Score, ROC-AUC
```

---

## 2. Metrik Regresi (*Regression Metrics*)

Kasus regresi bertujuan menebak target yang berupa **angka kontinu** (seperti harga rumah, temperatur udara, atau waktu pengiriman paket).

---

### 2.1 Mean Absolute Error (MAE)

#### A. Intuisi Konseptual
MAE adalah **rata-rata selisih jarak absolut** antara target sebenarnya dengan hasil tebakan:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n \left\vert y_i - \hat{y}_i \right\vert$$

* **Cara Membaca**: jika kamu memprediksi harga sewa rumah dan memperoleh $\text{MAE} = 250.000$, artinya rata-rata tebakan model meleset sebesar $\text{Rp}250.000$ dari harga sebenarnya, baik meleset lebih murah maupun lebih mahal.
* **Sifat Utama**: **tahan terhadap data pencilan (*robust to outliers*)**. Karena tidak ada operasi kuadrat, kesalahan tebakan yang besar tidak dihukum berlebihan. MAE mencerminkan performa tipikal pada sebagian besar data.

---

### 2.2 Mean Squared Error (MSE)

#### A. Intuisi Konseptual
MSE menghitung **rata-rata dari kuadrat selisih** antara target sebenarnya dengan hasil tebakan:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

* **Sifat Utama**: **sangat peka terhadap data pencilan (*sensitive to outliers*)**. Kesalahan kecil diberi hukuman ringan, tetapi kesalahan besar dihukum berkali-kali lipat secara eksponensial.
* **Kelemahan Sebagai Metrik Manusia**: satuannya ikut terkuadratkan. Jika target aslinya adalah rupiah ($\text{Rp}$), nilai MSE menjadi $\text{Rp}^2$ (rupiah kuadrat), sehingga maknanya sulit dipahami oleh pemangku kepentingan non-teknis.

---

### 2.3 Root Mean Squared Error (RMSE)

#### A. Intuisi Konseptual
RMSE adalah **akar kuadrat dari nilai MSE**:

$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

* **Mengapa RMSE Diciptakan?**: untuk menjembatani kelebihan MSE dengan kemudahan membaca satuan seperti MAE. Dengan menarik akar kuadrat, satuannya kembali normal ke satuan asli target (kembali menjadi $\text{Rp}$).
* **Perbandingan Intuitif MAE vs RMSE**:
  * Nilai RMSE **selalu lebih besar atau sama dengan** nilai MAE ($\text{RMSE} \ge \text{MAE}$).
  * Semakin jauh jarak perbedaan antara RMSE dan MAE pada satu model, artinya **semakin banyak kesalahan tebakan fatal (*outlier errors*)** yang dilakukan oleh model tersebut pada segelintir data.

```
CONTOH KASUS PENALTI GALAT:
Target Nyata: [100, 100, 100]
Tebakan Model A (Stabil)   : [110, 110, 110]  ---> MAE = 10,  RMSE = 10
Tebakan Model B (1 Meleset): [100, 100, 130]  ---> MAE = 10,  RMSE = 17.32

Meskipun nilai MAE kedua model sama-sama 10, RMSE Model B melonjak menjadi 17,32
karena menghukum keras satu tebakan yang meleset sebesar 30 poin.
```

---

### 2.4 Koefisien Determinasi ($R^2$ Score)

#### A. Intuisi Konseptual
MAE dan RMSE memiliki kelemahan: nilainya bergantung pada skala data. Meleset $10$ poin pada harga permen sangat fatal, tetapi meleset $10$ poin pada harga jet pribadi hampir tidak berarti.

$R^2$ Score (dibaca: *R-squared*) memecahkan masalah ini dengan mengukur **seberapa banyak persentase variasi pola data yang berhasil dijelaskan oleh model dibandingkan dengan model baseline paling bodoh (menebak rata-rata)**:

$$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

* $\text{SS}_{\text{res}}$: total kesalahan kuadrat dari model buatanmu.
* $\text{SS}_{\text{tot}}$: total kesalahan kuadrat jika kamu hanya menebak nilai rata-rata ($\bar{y}$).

#### B. Cara Menafsirkan Nilai $R^2$:
* **$R^2 = 1,0$ ($100\%$)**: model sempurna tanpa satu pun tebakan meleset.
* **$R^2 = 0,82$ ($82\%$)**: model berhasil menangkap $82\%$ pola perubahan data; sisanya $18\%$ merupakan derau acak atau faktor yang belum tercatat di data.
* **$R^2 = 0,0$**: model sama bodohnya dengan orang yang hanya menebak nilai rata-rata terus-menerus.
* **$R^2 < 0,0$ (Bernilai Negatif!)**: model bekerja **lebih buruk daripada tebakan rata-rata**. Ini adalah tanda bahaya bahwa model mengalami kerusakan fatal atau diterapkan pada data yang polanya berlawanan total.

---

### 2.5 Tabel Komparasi Metrik Regresi
| Metrik Evaluasi | Satuan Nilai | Sensitivitas Pencilan | Kapan Tepat Digunakan? |
| :--- | :--- | :--- | :--- |
| **MAE** | Sama dengan target | Rendah (*Robust*) | Saat dataset banyak kotoran derau atau pencilan liar |
| **MSE** | Kuadrat dari target | Sangat tinggi | Saat kesalahan besar membawa dampak bahaya fatal |
| **RMSE** | Sama dengan target | Tinggi | Pilihan standar industri jika penalti galat besar dibutuhkan |
| **$R^2$ Score** | Tanpa satuan ($-\infty$ s.d. $1$) | Tergantung data | Membandingkan performa antarmodel pada skala target berbeda |

---

### 2.6 Sumber Rujukan Topik Regresi
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning*, Edisi ke-2, Springer, Bab 2: “Overview of Supervised Learning”.
* **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**, *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2, Springer, Bab 2: “Assessing Model Accuracy”.

---

## 3. Metrik Klasifikasi (*Classification Metrics*)

Kasus klasifikasi bertujuan menebak **kategori diskrit** (misal: Transaksi Normal vs Penipuan, Pasien Sakit vs Sehat).

---

### 3.1 Intuisi Matriks Kebingungan (*Confusion Matrix*)
Sebelum menghitung persentase apa pun, seluruh hasil tebakan klasifikasi harus dipetakan ke dalam tabel silang 2x2 yang disebut **Confusion Matrix**.

Tabel ini memperlihatkan secara transparan di mana letak kebenaran dan di mana letak kebingungan model:

```
                          FAKTA SEBENARNYA (GROUND TRUTH)
                           KONDISI POSITIF       KONDISI NEGATIF
                        +---------------------+---------------------+
   TEBAKAN MODEL        |                     |                     |
      POSITIF           | TRUE POSITIVE (TP)  | FALSE POSITIVE (FP) |
                        | Tebak Positif,      | Tebak Positif,      |
                        | Nyatanya Benar      | Nyatanya Salah      |
                        |                     | (Alarm Palsu)       |
                        +---------------------+---------------------+
   TEBAKAN MODEL        |                     |                     |
      NEGATIF           | FALSE NEGATIVE (FN) | TRUE NEGATIVE (TN)  |
                        | Tebak Negatif,      | Tebak Negatif,      |
                        | Nyatanya Salah      | Nyatanya Benar      |
                        | (Luput / Lolos)     |                     |
                        +---------------------+---------------------+
```

#### Empat Komponen Inti:
1. **True Positive (TP)**: fakta asli Positif, model menebak Positif. *(Tepat sasaran)*.
2. **True Negative (TN)**: fakta asli Negatif, model menebak Negatif. *(Tepat menolak)*.
3. **False Positive (FP) — Galat Tipe I**: fakta asli Negatif, tetapi model menebak Positif. Sering disebut **Alarm Palsu**.
   * *Contoh*: email penting teman masuk ke folder spam; orang sehat divonis sakit.
4. **False Negative (FN) — Galat Tipe II**: fakta asli Positif, tetapi model menebak Negatif. Sering disebut **Kondisi Luput**.
   * *Contoh*: pasien penderita kanker ganas dibilang sehat dan disuruh pulang; transaksi pembobolan rekening dibiarkan lolos.

---

### 3.2 Akurasi (*Accuracy*) dan Mengapa Akurasi Sering Menipu

#### A. Definisi
Akurasi adalah proporsi seluruh tebakan yang benar (positif maupun negatif) dibagi dengan total seluruh data:

$$\text{Akurasi} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}}$$

#### B. Paradoks Akurasi pada Data Timpang (*The Accuracy Paradox*)
Bayangkan kamu membangun model deteksi penyakit langka yang hanya diderita oleh $1$ dari $1.000$ orang ($0,1\%$ penderita, $99,9\%$ orang sehat):
* Kamu membuat model pemalas yang tidak belajar apa pun. Model ini diprogram untuk **selalu menebak "SEHAT" kepada semua orang**.
* Dari $1.000$ pasien:
  * Model menebak $999$ orang sehat dengan benar ($\text{TN} = 999$).
  * Model gagal mendeteksi $1$ orang yang sakit ($\text{FN} = 1$).
* **Skor Akurasi**: $\frac{999}{1.000} = 99,9\%$.

Akurasi model terlihat luar biasa tinggi ($99,9\%$), namun model tersebut **sama sekali tidak berguna** karena pasien yang benar-benar sakit justru luput dan terancam meninggal. 

> **Kaidah Utama**: jangan pernah mengandalkan Akurasi jika sebaran kelas pada datasetmu tidak seimbang (*class imbalance*).

---

### 3.3 Presisi (*Precision*) vs Ingatan (*Recall*)

Ketika akurasi tidak bisa dipercaya, kita menggunakan dua sudut pandang yang berbeda: Presisi dan Recall.

```
                  DUA SUDUT PANDANG KUALITAS TEBAKAN
                                     |
        +----------------------------+----------------------------+
        |                                                         |
        v                                                         v
     PRESISI (PRECISION)                                 INGATAN (RECALL)
 Fokus pada Kualitas Tebakan Positif                Fokus pada Kelengkapan Penemuan
 "Dari semua yang dituduh Positif,                  "Dari semua yang aslinya Positif,
  berapa yang tuduhannya benar?"                     berapa banyak yang berhasil dijaring?"
 Target: Meminimalkan False Positive (Alarm Palsu)  Target: Meminimalkan False Negative (Luput)
```

#### A. Presisi (*Precision*)
$$\text{Presisi} = \frac{\text{TP}}{\text{TP} + \text{FP}}$$

* **Pertanyaan Kunci**: *"Ketika model menebak kelas Positif, seberapa besar peluang tebakan tersebut dapat dipercaya?"*
* **Kapan Mengutamakan Presisi?**: ketika biaya atau kerugian akibat **Alarm Palsu (False Positive)** sangat mahal.
  * *Kasus Filter Spam Email*: lebih baik membiarkan 1 email spam lolos ke kotak masuk daripada salah menandai 1 email tawaran kerja penting sebagai spam dan menghapusnya.
  * *Rekomendasi Video YouTube*: video yang disodorkan harus benar-benar disukai pengguna agar pengguna tidak kecewa dan menutup aplikasi.

#### B. Ingatan (*Recall / Sensitivity*)
$$\text{Recall} = \frac{\text{TP}}{\text{TP} + \text{FN}}$$

* **Pertanyaan Kunci**: *"Dari seluruh kasus positif yang bertebaran di dunia nyata, berapa persen yang berhasil ditemukan oleh model?"*
* **Kapan Mengutamakan Recall?**: ketika biaya atau risiko akibat **Kasus Luput (False Negative)** berakibat fatal.
  * *Diagnosis Penyakit Kritis*: lebih aman salah mencurigai orang sehat untuk diperiksa ulang di laboratorium daripada meloloskan pasien kanker tanpa perawatan.
  * *Deteksi Penipuan Finansial*: bank memilih menghentikan sementara transaksi yang sedikit mencurigakan untuk konfirmasi daripada membiarkan uang nasabah terkuras habis.

---

### 3.4 Tarik-Ulur Presisi dan Recall (*Trade-off*)
Secara alami, Presisi dan Recall saling bertolak belakang:
* Jika kamu ingin model menjaring semua kasus positif tanpa ada yang luput (memaksimalkan Recall), model akan menjadi sangat penakut dan banyak membunyikan alarm palsu (Presisi anjlok).
* Jika kamu ingin model hanya berani menebak positif jika ia yakin 100% (memaksimalkan Presisi), model akan menjadi sangat pemilih dan membiarkan banyak kasus positif lolos begitu saja (Recall anjlok).

---

### 3.5 F1-Score: Keseimbangan Harmonis

Jika kamu ingin satu angka ringkas yang menyeimbangkan antara Presisi dan Recall, gunakan **F1-Score**. F1-Score dihitung menggunakan **rata-rata harmonik**, bukan rata-rata biasa:

$$\text{F1-Score} = 2 \times \frac{\text{Presisi} \times \text{Recall}}{\text{Presisi} + \text{Recall}}$$

#### Mengapa Wajib Rata-Rata Harmonik?
Perhatikan perbandingannya dengan rata-rata aritmatika biasa jika sebuah model bermain curang dengan memprediksi positif untuk semua data ($\text{Recall} = 1,0$ dan $\text{Presisi} = 0,1$):
* **Rata-rata biasa**: $\frac{1,0 + 0,1}{2} = 0,55$ *(Menipu, terlihat cukup lumayan)*.
* **Rata-rata harmonik (F1)**: $2 \times \frac{1,0 \times 0,1}{1,0 + 0,1} = \frac{0,2}{1,1} \approx 0,18$ *(Jujur, nilai langsung anjlok)*.

Rata-rata harmonik memastikan bahwa jika salah satu komponen (baik Presisi maupun Recall) sangat buruk, nilai F1-Score akan langsung jatuh mendekati nilai yang terendah. Model hanya akan mendapat F1-Score tinggi jika **kedua metrik sama-sama bagus**.

---

### 3.6 Sumber Rujukan Topik Klasifikasi
* **Powers, David M. W. (2011)**, “Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation”, *Journal of Machine Learning Technologies*, 2(1), hlm. 37–63.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 3: “Classification (Precision/Recall Trade-off)”.
* **Fawcett, Tom (2006)**, “An introduction to ROC analysis”, *Pattern Recognition Letters*, 27(8), hlm. 861–874.

---

## 4. Panduan Memilih Metrik yang Tepat (*Decision Guide*)

Gunakan bagan keputusan di bawah ini untuk menentukan metrik mana yang paling relevan dengan prioritas bisnismu:

```
                               [ IDENTIFIKASI JENIS MASALAH ]
                                              |
                +-----------------------------+-----------------------------+
                |                                                           |
                v                                                           v
       [ KASUS REGRESI ]                                           [ KASUS KLASIFIKASI ]
        (Menebak Angka)                                              (Menebak Kategori)
                |                                                           |
        +-------+-------+                                           +-------+-------+
        |               |                                           |               |
Apakah banyak data    Ingin menghukum                             Apakah sebaran   Dataset sangat
pencilan ekstrem?     galat besar?                                kelas seimbang?  timpang / langka?
        |               |                                           |               |
     [ YA ]          [ YA ]                                      [ YA ]          [ YA ]
        v               v                                           v               v
     PILIH MAE      PILIH RMSE                                PILIH AKURASI         |
                                                                                    |
                                            +---------------------------------------+
                                            |
                                 Mana risiko yang paling berbahaya?
                                            |
                         +------------------+------------------+
                         |                                     |
                Alarm Palsu Fatal?                      Kasus Luput Fatal?
              (False Positive Bahaya)                 (False Negative Bahaya)
                         |                                     |
                         v                                     v
                  PILIH PRESISI                           PILIH RECALL
                         \                                     /
                          +-----------------+-----------------+
                                            |
                                  Ingin keseimbangan keduanya?
                                            |
                                            v
                                      PILIH F1-SCORE
```

---

## 5. Ringkasan Kasus Nyata di Dunia Industri

| Domain Masalah | Karakteristik Data | Metrik Prioritas | Alasan Pertimbangan Bisnis |
| :--- | :--- | :--- | :--- |
| **Deteksi Kanker Medis** | Positif sangat langka ($<1\%$) | **Recall** | Pasien sakit tidak boleh luput karena berisiko kematian |
| **Penyaring Surel Spam** | Seimbang / Sedang | **Presisi** | Surel penting tidak boleh salah masuk ke tong sampah |
| **Estimasi Tarif Ojek Online** | Banyak lonjakan cuaca/macet | **MAE / RMSE** | Tarif harus wajar dan stabil bagi mitra dan pelanggan |
| **Deteksi Transaksi Fraud** | Penipuan sangat langka ($<0,1\%$) | **F1 / PR-AUC** | Menyeimbangkan antara keamanan rekening dan kenyamanan nasabah |

---

## 6. Daftar Pustaka Terverifikasi

1. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
2. **James, Gareth, Witten, Daniela, Hastie, Trevor, dan Tibshirani, Robert (2021)**. *An Introduction to Statistical Learning: with Applications in R*, Edisi ke-2. Springer.
3. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
4. **Fawcett, Tom (2006)**. “An introduction to ROC analysis”. *Pattern Recognition Letters*, 27(8), hlm. 861–874.
5. **Powers, David M. W. (2011)**. “Evaluation: From Precision, Recall and F-Measure to ROC, Informedness, Markedness & Correlation”. *Journal of Machine Learning Technologies*, 2(1), hlm. 37–63.
6. **Murphy, Kevin P. (2012)**. *Machine Learning: A Probabilistic Perspective*. The MIT Press.
7. **Sokolova, Marina, dan Lapalme, Guy (2009)**. “A systematic analysis of performance measures for classification tasks”. *Information Processing & Management*, 45(4), hlm. 427–437.