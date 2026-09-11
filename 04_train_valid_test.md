# Modul Pembelajaran: Train, Validation, dan Test Split

Modul pembelajaran ini membedah prinsip paling mendasar dalam rekayasa *Machine Learning* (ML): **bagaimana cara membagi data secara benar agar model tidak menipu kita dengan hasil pintar yang palsu**. Di sini kita akan mempelajari pembagian data latih, validasi, dan uji, bahaya kebocoran data (*data leakage*), teknik validasi silang (*cross-validation*), serta aturan ketat mengapa data uji hampir tidak pernah boleh disentuh.

---

## Glosarium Istilah Penting
Agar tidak membingungkan, berikut arti istilah teknis yang akan sering digunakan dalam modul ini:
* **Fitur (*Features*)**: variabel masukan atau data karakteristik yang dipakai komputer untuk membuat perkiraan (misal: luas tanah, jumlah kamar, lokasi).
* **Target yang Ingin Diprediksi (*Target / Label*)**: jawaban atau hasil akhir yang ingin ditebak oleh model (misal: harga rumah).
* **Hasil Tebakan (*Predictions / Output*)**: jawaban yang dikeluarkan oleh model setelah memproses fitur masukan.
* **Parameter Dalam (*Model Weights*)**: setelan di dalam tubuh model yang angkanya dicari dan diperbaiki sendiri oleh algoritma saat proses belajar berlangsung.
* **Pengaturan Luar (*Hyperparameters*)**: setelan tombol atau konfigurasi di luar model yang ditentukan secara manual oleh manusia sebelum latihan dimulai (misalnya: tingkat kedalaman pohon keputusan atau kecepatan belajar).
* **Data Asing (*Unseen Data*)**: data baru di dunia nyata yang sama sekali belum pernah dilihat atau dipelajari oleh model selama proses latihan.
* **Menghafal Mati (*Overfitting*)**: kondisi buruk di mana model hafal luar kepala setiap baris data latihannya beserta gangguan datanya, tetapi langsung bingung dan salah tebak saat disodori data baru.

---

## 1. Mengapa Pembagian Data Itu Penting? (*Why Splitting Data Matters*)

Masalah paling umum bagi pemula di bidang Machine Learning adalah menguji kepintaran model menggunakan data yang sama persis dengan data yang dipakai untuk melatihnya.

### 1.1 Analogi Ujian Sekolah
Bayangkan seorang guru matematika memberikan 50 butir soal latihan kepada muridnya beserta kunci jawabannya pada hari Senin. Pada hari Jumat, guru tersebut mengadakan ujian akhir dan memberikan **50 butir soal yang sama persis** dengan angka yang tidak diubah sedikit pun.

Jika murid tersebut mendapat nilai 100, apakah guru tersebut bisa menjamin sang murid benar-benar paham konsep matematika? **Tentu tidak.** Sang murid bisa saja hanya menghafal mati urutan huruf kunci jawaban tanpa memahami cara perhitungannya.

```
SKENARIO SALAH (Tanpa Pembagian Data):
  [ 100% Seluruh Data ] ---> Dipakai Belajar (Latihan)
            |
            v
  [ 100% Data yang Sama ] ---> Diuji Kembali
            |
            v
  Hasil: Akurasi 99% (Palsu / Menipu!)
  Saat dipakai di dunia nyata: Akurasi anjlok menjadi 50%.
```

### 1.2 Bahaya Optimisme Semu (*Optimism Bias*)
Jika kita melatih dan menguji model pada dataset yang sama, evaluasi tersebut akan menghasilkan **optimisme semu**—sebuah ilusi bahwa model kita sangat cerdas. 

Di dunia nyata, model dibuat bukan untuk menebak hal yang sudah lewat, melainkan untuk **memprediksi masa depan atau data baru yang belum pernah dilihat (*unseen data*)**. Membagi data adalah satu-satunya cara ilmiah untuk mengukur kemampuan **generalisasi** model secara jujur sebelum model dilepas ke peladen produksi (*production*).

---

### 1.3 Sumber Rujukan Topik 1
* **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2, Springer, Bab 7: “Model Assessment and Selection”.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “Testing and Validating”.

---

## 2. Train vs Validation vs Test Sets (Tiga Pembagian Utama)

Untuk membangun sistem Machine Learning yang kokoh, dataset mentah idealnya dipecah menjadi **tiga bagian terpisah** dengan fungsi yang berbeda total.

```
                          TOTAL SELURUH DATASET
  +-------------------------------------+------------------+------------------+
  |                                     |                  |                  |
  |          DATA LATIH (TRAIN)         | DATA VALIDASI    |  DATA UJI (TEST) |
  |              (~70% - 80%)           |   (~10% - 15%)   |   (~10% - 15%)   |
  |                                     |                  |                  |
  +-------------------------------------+------------------+------------------+
                   |                              |                  |
                   v                              v                  v
             Dipakai untuk:                 Dipakai untuk:     Dipakai untuk:
          - Model belajar pola           - Memilih model    - Penilaian akhir
          - Memperbarui parameter          terbaik            sebelum rilis
            dalam (bobot)                - Menyetel tombol  - Mengukur performa
                                           luar (tuning)      jujur di dunia nyata
```

---

### 2.1 Rincian Fungsi Masing-Masing Bagian

#### A. Data Latih (*Training Set*)
* **Analogi**: buku paket materi dan soal latihan harian di kelas.
* **Fungsi**: data utama yang dibaca langsung oleh algoritma untuk mencari rumus matematika dan menyesuaikan nilai parameter dalam (*weights*) agar tebakannya semakin mendekati target yang ingin diprediksi.
* **Porsi Umum**: biasanya memakan porsi terbesar, sekitar $70\%$ hingga $80\%$ dari total data.

#### B. Data Validasi (*Validation Set / Dev Set*)
* **Analogi**: ujian simulasi atau *tryout* mingguan sebelum ujian akhir.
* **Fungsi**: digunakan oleh **manusia (perekayasa ML)** untuk membandingkan beberapa algoritma sekaligus (misalnya membandingkan Pohon Keputusan vs Regresi Logistik), serta mencari setelan tombol luar (*hyperparameter tuning*) yang paling pas.
* Model **tidak pernah belajar langsung** dari data ini, tetapi kita menggunakan skor evaluasi data validasi untuk memutuskan arah perbaikan model dan mendeteksi apakah model mulai menghafal mati (*overfitting*).
* **Porsi Umum**: sekitar $10\%$ hingga $15\%$ dari total data.

#### C. Data Uji (*Test Set*)
* **Analogi**: naskah ujian nasional resmi yang disegel di dalam brankas.
* **Fungsi**: memberikan penilaian akhir yang sepenuhnya objektif dan tidak memihak mengenai seberapa andal model saat disodori data baru di dunia nyata.
* Data ini **sama sekali tidak boleh dilihat** oleh model maupun manusia selama proses eksperimen dan penyesuaian setelan model berlangsung.
* **Porsi Umum**: sekitar $10\%$ hingga $15\%$ dari total data.

---

### 2.2 Bagaimana Rasio Pembagian pada Era Data Raksasa (*Big Data*)?
Rasio klasik $70:15:15$ atau $80:10:10$ sangat cocok jika jumlah total data kita berkisar antara ribuan hingga ratusan ribu baris.

Namun, di era kecerdasan buatan modern dengan jutaan baris data (misal: 10.000.000 data):
* Kita tidak memerlukan $10\%$ (1.000.000 data) hanya untuk data uji.
* Mengambil $100.000$ data saja ($1\%$) sudah lebih dari cukup untuk mendapatkan evaluasi yang valid secara statistik.
* Oleh karena itu, pada skala data raksasa, rasionya bergeser menjadi **$98:1:1$** atau bahkan **$99:0,5:0,5$** (Andrew Ng, *Deep Learning Specialization*).

---

### 2.3 Tabel Komparasi Train vs Validation vs Test Sets
| Dimensi Evaluasi | Data Latih (*Train Set*) | Data Validasi (*Validation Set*) | Data Uji (*Test Set*) |
| :--- | :--- | :--- | :--- |
| **Pengguna Utama** | Algoritma (komputer) | Perekayasa perangkat lunak (manusia) | Penilai akhir / Manajer produk |
| **Tujuan Pemakaian** | Belajar pola dari nol | Menyetel tombol luar & pilih model | Mengukur performa jujur sebelum rilis |
| **Frekuensi Sentuh** | Berulang-ulang di setiap putaran (*epoch*) | Sering (setiap selesai satu eksperimen) | **Hanya satu kali** di ujung proyek |
| **Pengaruh ke Bobot Model** | Mengubah parameter dalam secara langsung | Mengubah keputusan setelan secara tidak langsung | **Nol** (tidak boleh memengaruhi model) |

---

### 2.4 Sumber Rujukan Topik 2
* **Ng, Andrew (2018)**, *Machine Learning Yearning: Technical Strategy for AI Engineers in the Era of Deep Learning*, Bab 5: “Your development and test sets”.
* **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**, *Deep Learning*, MIT Press, Bab 5: “Machine Learning Basics (Hyperparameters and Validation Sets)”.

---

## 3. Kebocoran Data (*Data Leakage*) — SANGAT PENTING

Kebocoran data (*data leakage*) adalah salah satu kesalahan paling berbahaya dalam rekayasa Machine Learning yang sering membuat model terlihat hebat di laboratorium, tetapi langsung gagal total saat dipakai oleh pengguna nyata.

### 3.1 Definisi dan Analogi
**Kebocoran Data** adalah situasi di mana informasi dari luar data latih (khususnya contekan dari data uji atau informasi masa depan) tanpa sengaja merembes masuk ke dalam proses belajar model.

* **Analogi Contekan**: seorang murid berhasil mencuri selembar kertas kunci jawaban ujian sebelum hari H. Saat mengerjakan soal latihan, dia mencocokkan rumus dengan kunci jawaban curian tersebut. Hasil latihannya terlihat sempurna, tetapi kepintarannya palsu karena terbantu contekan.

---

### 3.2 Tiga Bentuk Kebocoran Data yang Paling Sering Terjadi

#### A. Kebocoran Tahap Persiapan Data (*Preprocessing Leakage*)
Ini adalah bentuk kebocoran yang paling sering dilakukan tanpa sadar.
* **Kesalahan Fatal**: melakukan standardisasi skala data (misalnya menghitung rata-rata dan deviasi standar) atau mengisi nilai kosong (*imputation*) **pada seluruh dataset sekaligus sebelum data dipotong** menjadi data latih dan data uji.
* **Mengapa Bocor?**: nilai rata-rata dari data uji ikut dihitung dan memberi petunjuk terselubung ke dalam data latih.
* **Urutan yang Benar**:
  ```
  URUTAN KELIRU (BOCOR):
  [ Semua Data ] ---> Hitung Rata-rata & Standarisasi ---> Bagi ke Train & Test
  
  URUTAN BENAR (BEBAS BOCOR):
  [ Semua Data ] ---> Bagi ke Data Latih & Data Uji
                             |
                             +---> Hitung Rata-rata HANYA dari Data Latih!
                             |
                             +---> Terapkan angka rata-rata tersebut ke Data Uji
  ```

#### B. Kebocoran Dimensi Waktu (*Temporal Leakage*)
Terjadi pada data yang memiliki urutan waktu (misal: harga saham harian, riwayat transaksi belanja, atau prakiraan cuaca).
* **Kesalahan Fatal**: membagi data secara acak menggunakan fungsi bawaan (*random train-test split*).
* **Mengapa Bocor?**: model menggunakan data transaksi hari Rabu dan Jumat untuk menebak transaksi hari Kamis. Model secara tidak langsung "melihat masa depan" untuk menebak masa lalu.
* **Urutan yang Benar**: gunakan pemotongan kronologis (*chronological split*). Misal: data bulan Januari hingga Oktober untuk data latih, data bulan November untuk validasi, dan data bulan Desember untuk uji.

#### C. Kebocoran Data Duplikat / Identitas (*Group / Identity Leakage*)
Terjadi jika beberapa baris data berasal dari subjek atau orang yang sama.
* **Contoh Kasus**: dalam diagnosis tumor paru-paru dari foto rontgen, satu pasien memiliki 5 foto rontgen dari sudut berbeda. Jika data diacak, 3 foto pasien tersebut masuk ke data latih dan 2 foto masuk ke data uji.
* **Mengapa Bocor?**: model tidak belajar pola penyakit, melainkan mengenali bentuk anatomi atau ciri fisik unik pasien tersebut yang sudah dihafal dari data latih.

---

### 3.3 Tanda-Tanda Terjadinya Kebocoran Data
Curigailah proses kerja jika menemukan gejala-gejala berikut:
1. Akurasi model terlihat terlalu indah untuk menjadi kenyataan (misalnya langsung mencapai $99,8\%$ pada percobaan pertama untuk masalah yang sangat rumit).
2. Terdapat satu fitur masukan yang korelasi atau bobot kepentingannya (*feature importance*) luar biasa tinggi dan tidak masuk akal.
3. Model bekerja luar biasa sempurna pada data latih dan data uji di komputer, tetapi saat sistem dihubungkan ke data nyata di server, tingkat kesalahannya melonjak sangat tinggi.

---

### 3.4 Sumber Rujukan Topik 3
* **Kaufman, Shachar, Rosset, Saharon, dan Perlich, Claudia (2012)**, “Leakage in Data Mining: Formulation, Detection, and Avoidance”, *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), hlm. 1–21.
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 2: “Prepare the Data for Machine Learning Algorithms”.

---

## 4. Validasi Hold-Out (*Hold-Out Validation*)

Validasi *Hold-Out* adalah metode paling sederhana dalam menguji model.

```
                    DATASET ASLI
  +---------------------------------------+-------------------+
  |           DATA LATIH (TRAIN)          |  DATA UJI (HOLD)  |
  |                  80%                  |        20%        |
  +---------------------------------------+-------------------+
```

### 4.1 Cara Kerja
Kita mengambil dataset asli, lalu "menahan" (*hold out*) sebagian kecil data (misalnya $20\%$) dan menyimpannya di tempat terpisah sebagai bahan evaluasi. Sisanya ($80\%$) dipakai untuk melatih model.

### 4.2 Kelebihan dan Kelemahan
* **Kelebihan**:
  * Sangat sederhana untuk dipahami dan ditulis dalam kode program.
  * Sangat hemat waktu dan ringan komputasi karena model hanya perlu dilatih **satu kali**.
* **Kelemahan Fatal (*Sampling Bias*)**:
  * Hasil akurasi sangat bergantung pada faktor keberuntungan: data mana yang kebetulan terpotong masuk ke data latih dan data mana yang masuk ke data uji.
  * Jika dataset berukuran kecil (misal hanya 200 data), membuang 40 data ke tempat uji akan membuat data belajar menjadi terlalu sedikit. Jika di antara 40 data uji tersebut kebetulan terdapat banyak kasus langka, nilai akurasi model bisa terlihat anjlok drastis padahal aslinya model cukup bagus.

---

### 4.3 Sumber Rujukan Topik 4
* **Mitchell, Tom M. (1997)**, *Machine Learning*, McGraw-Hill Computer Science Series, Bab 5: “Evaluating Hypotheses”.
* **Kohavi, Ron (1995)**, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”, *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), hlm. 1137–1145.

---

## 5. Cross-Validation / Validasi Silang (Konsep Tanpa Rumus Rumit)

Jika dataset kita berukuran terbatas, mengandalkan pemotongan satu kali ala *Hold-Out* sangat berisiko. Solusi standarnya adalah menggunakan **Validasi Silang (*Cross-Validation*)**, khususnya metode paling populer yang disebut **k-Fold Cross-Validation**.

---

### 5.1 Cara Kerja k-Fold Cross-Validation
Prinsip dasar metode ini adalah **sistem piket atau giliran ujian secara adil**:

1. Tentukan angka $k$, misalnya $k = 5$.
2. Dataset dibagi rata menjadi 5 kelompok data berukuran sama yang disebut **Lipatan (*Fold*)**.
3. Proses pelatihan dan pengujian diulang sebanyak **5 putaran**:
   * **Putaran 1**: Fold 1 dijadikan data uji/validasi. Fold 2, 3, 4, dan 5 dipakai untuk belajar.
   * **Putaran 2**: Fold 2 dijadikan data uji/validasi. Fold 1, 3, 4, dan 5 dipakai untuk belajar.
   * **Putaran 3**: Fold 3 dijadikan data uji/validasi. Fold 1, 2, 4, dan 5 dipakai untuk belajar.
   * **Putaran 4**: Fold 4 dijadikan data uji/validasi. Fold 1, 2, 3, dan 5 dipakai untuk belajar.
   * **Putaran 5**: Fold 5 dijadikan data uji/validasi. Fold 1, 2, 3, dan 4 dipakai untuk belajar.
4. Di akhir, kita memiliki 5 nilai skor akurasi. Kita hitung **rata-rata** dan rentang penyimpangannya (*standard deviation*).

```
ILUSTRASI 5-FOLD CROSS-VALIDATION:

Putaran 1: [ UJI ] [ LATIH ] [ LATIH ] [ LATIH ] [ LATIH ] ---> Skor Akurasi: 85%
Putaran 2: [ LATIH ] [ UJI ] [ LATIH ] [ LATIH ] [ LATIH ] ---> Skor Akurasi: 87%
Putaran 3: [ LATIH ] [ LATIH ] [ UJI ] [ LATIH ] [ LATIH ] ---> Skor Akurasi: 83%
Putaran 4: [ LATIH ] [ LATIH ] [ LATIH ] [ UJI ] [ LATIH ] ---> Skor Akurasi: 86%
Putaran 5: [ LATIH ] [ LATIH ] [ LATIH ] [ LATIH ] [ UJI ] ---> Skor Akurasi: 84%

HASIL AKHIR: Rata-rata Skor = 85% (+/- 1.4%)
Setiap baris data pernah merasakan menjadi murid (belajar) dan menjadi lembar ujian!
```

---

### 5.2 Dua Variasi Khusus yang Wajib Diketahui

#### A. Stratified k-Fold (Validasi Proporsional)
* **Kapan Dipakai?**: saat target yang ingin diprediksi jumlahnya sangat timpang (*imbalanced data*). Contoh: kasus penipuan kartu kredit di mana transaksi sah ada $99\%$ dan penipuan hanya $1\%$.
* **Cara Kerja**: metode ini memastikan bahwa di setiap lipatan (*fold*), perbandingan persentase antara kelas normal dan kelas langka **selalu sama persis** dengan perbandingan di dataset asli. Hal ini mencegah adanya lipatan uji yang tidak kebagian sampel penipuan sama sekali.

#### B. Time Series Split (Validasi Urutan Waktu)
* **Kapan Dipakai?**: untuk data yang terikat garis waktu (saham, penjualan musiman).
* **Cara Kerja**: data tidak boleh diacak bolak-balik. Pelatihan selalu menggunakan data masa lalu dan pengujian selalu dilakukan pada data satu langkah di masa depan (*walk-forward testing*).

---

### 5.3 Kapan Memilih Hold-Out vs Cross-Validation?
* **Pilih Cross-Validation jika**: dataset berukuran kecil hingga sedang (ratusan hingga puluhan ribu baris data) dan kita membutuhkan kepastian performa yang sangat stabil tanpa bias pemotongan acak.
* **Pilih Hold-Out jika**: dataset berukuran masif (jutaan baris) atau model memerlukan waktu berhari-hari untuk sekali latihan (misalnya model *Deep Learning* raksasa), karena mengulang latihan sebanyak $k$ kali akan memakan biaya server yang terlampau mahal.

---

### 5.4 Sumber Rujukan Topik 5
* **Kohavi, Ron (1995)**, “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”, *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), hlm. 1137–1145.
* **Bishop, Christopher M. (2006)**, *Pattern Recognition and Machine Learning*, Springer, Bab 1: “Introduction (Model Selection)”.

---

## 6. Kapan Data Uji Boleh Disentuh? (Hampir Tidak Pernah!)

Di industri rekayasa Machine Learning terdapat sebuah aturan emas yang pantang dilanggar: **Data Uji adalah brankas terlarang (*The Locked Safe*)**.

### 6.1 Mengapa Tidak Boleh Disentuh Berulang Kali?
Sering kali pemula tergoda melakukan alur kerja yang salah seperti ini:
1. Membagi data menjadi data latih dan data uji.
2. Melatih model di data latih, lalu mengukurnya di data uji.
3. Melihat hasilnya di data uji kurang memuaskan (misal hanya $75\%$).
4. Mengubah setelan tombol model (*tuning*), lalu mengujinya lagi ke data uji.
5. Mengulang siklus ini berkali-kali sampai angka di data uji mencapai $92\%$.

**Apa yang salah dari alur di atas?**
Meskipun model tidak belajar langsung dari data uji, **otak sang perekayasa perangkat lunak telah menyesuaikan setelan model agar cocok dengan karakteristik data uji tersebut**. Fenomena ini disebut **Kebocoran Terselubung (*Data Snooping Bias*)**. Data uji tersebut kini sudah terkontaminasi dan tidak lagi menjadi bahan ujian yang objektif.

---

### 6.2 Kapan Tepatnya Data Uji Boleh Disentuh?
Jawabannya sangat tegas: **Tepat satu kali saja di akhir siklus proyek**, yaitu ketika:
* Eksperimen pemilihan berbagai model sudah selesai tuntas.
* Penyetelan tombol luar (*hyperparameter tuning*) menggunakan data validasi sudah selesai.
* Kamu sudah yakin model terbaik sudah terpilih dan ingin mengetahui estimasi kinerja jujurnya sebelum sistem dikirim ke klien atau dipasang di peladen produksi.

### 6.3 Bagaimana Jika Skor di Data Uji Ternyata Jelek?
Jika saat data uji dibuka untuk pertama kali dan hasilnya ternyata sangat buruk:
* **JANGAN** mengubah-ubah setelan model agar cocok dengan data uji tersebut!
* Itu adalah sinyal peringatan bahwa proses validasi kamu sebelumnya sudah bocor atau salah asumsi.
* Yang harus dilakukan: kembali ke meja kerja, perbaiki kualitas pembersihan data, buat fitur baru yang lebih relevan, kumpulkan data baru yang lebih segar, atau evaluasi ulang model **hanya menggunakan data validasi yang baru**.

---

### 6.4 Sumber Rujukan Topik 6
* **Géron, Aurélien (2022)**, *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3, O'Reilly Media, Bab 1: “Data Snooping Bias”.
* **Zinkevich, Martin**, “Rules of Machine Learning: Best Practices for ML Engineering”, Google Research, Aturan #40: “Keep ensembles simple”.

---

## 7. Kerangka Kerja Praktis Alur Pembagian Data

Diagram alur kerja baku yang aman dari kebocoran data untuk diterapkan pada proyek nyata:

```
                            [ DATASET MENTAH ]
                                     |
             +-----------------------+-----------------------+
             |                                               |
             v                                               v
    [ DATA LATIH + VALIDASI ]                         [ DATA UJI (TEST) ]
             (80%)                                           (20%)
             |                                                 |
             v                                                 |
   +--------------------+                                      |
   | PEMBERSIHAN DATA   |                                      |
   | (Scaling, Impute)  |                                      |
   +--------------------+                                      |
             |                                                 |
             v                                                 |
   +--------------------+                                      |
   | K-FOLD VALIDASI    |                                      |
   | - Eksperimen Model |                                      |
   | - Setel Tombol     |                                      |
   +--------------------+                                      |
             |                                                 |
             v                                                 v
   [ MODEL TERBAIK FINAL ] ----------------------------> [ BUKA BRANKAS ]
                                                         (Uji Hanya 1 Kali)
                                                               |
                                                               v
                                                    [ PERFORMA NYATA FINAL ]
                                                               |
                                                               v
                                                    [ RILIS KE PRODUKSI ]
```

---

## 8. Daftar Pustaka Terverifikasi

1. **Hastie, Trevor, Tibshirani, Robert, dan Friedman, Jerome (2009)**. *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, Edisi ke-2. Springer Science & Business Media.
2. **Goodfellow, Ian, Bengio, Yoshua, dan Courville, Aaron (2016)**. *Deep Learning*. MIT Press.
3. **Géron, Aurélien (2022)**. *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow*, Edisi ke-3. O'Reilly Media.
4. **Kohavi, Ron (1995)**. “A Study of Cross-Validation and Bootstrap for Accuracy Estimation and Model Selection”. *International Joint Conference on Artificial Intelligence (IJCAI)*, 14(2), hlm. 1137–1145.
5. **Kaufman, Shachar, Rosset, Saharon, dan Perlich, Claudia (2012)**. “Leakage in Data Mining: Formulation, Detection, and Avoidance”. *ACM Transactions on Knowledge Discovery from Data (TKDD)*, 6(4), hlm. 1–21.
6. **Ng, Andrew (2018)**. *Machine Learning Yearning: Technical Strategy for AI Engineers in the Era of Deep Learning*. deeplearning.ai.
7. **Mitchell, Tom M. (1997)**. *Machine Learning*. McGraw-Hill Education.
8. **Bishop, Christopher M. (2006)**. *Pattern Recognition and Machine Learning*. Springer.
9. **Zinkevich, Martin**. “Rules of Machine Learning: Best Practices for ML Engineering”. Google Research.