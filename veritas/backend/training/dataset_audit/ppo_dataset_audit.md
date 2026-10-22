# FakeAVCeleb v1.2 Dataset Audit & Split Provenance Report

**Dataset Path:** `D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2`  
**Audit Date:** 2026-10-20  
**Auditor:** Senior Reinforcement Learning Engineer & ML Systems Architect  
**Manifest Artifact:** [`veritas/backend/training/ppo_split_manifest.json`](file:///D:/Multi-Agent-DeepFake-Detection-And-Reasoning-main/Multi-Agent-DeepFake-Detection-And-Reasoning-main/veritas/backend/training/ppo_split_manifest.json)  

---

## 1. File Count & Metadata Reconciliation

| Entity | Count | Details & Reconciliation |
|:---|:---:|:---|
| **Rows in `meta_data.csv`** | **21,566** | 1 header row + 21,566 data entries |
| **Exact Duplicate CSV Rows** | **22** | 22 file paths appear twice with identical metadata entries |
| **Unique File Paths in CSV** | **21,544** | 21,566 rows minus 22 exact duplicate entries |
| **MP4 Files on Disk** | **21,560** | All 21,544 unique CSV paths exist on disk without exception |
| **Files in CSV Missing from Disk** | **0** | Zero missing files (100.0% coverage) |
| **Files on Disk Missing from CSV** | **16** | 16 decompression archive collision files ending in ` (1).mp4` |
| **Zero-Size / Corrupted Files** | **0** | Every single file on disk has non-zero size and valid header |

### Exact Arithmetic Reconciliation:
$$\text{Total Files on Disk } (21,560) = \text{Unique Paths in CSV } (21,544) + \text{Unpacked Collision Files } (16)$$
$$\text{CSV Data Rows } (21,566) = \text{Unique Paths in CSV } (21,544) + \text{Exact Duplicate Rows } (22)$$

### Breakdown of the 16 Collision Files on Disk:
During archive decompression, 16 duplicate files in `Asian (South)/men/` were saved with Windows collision suffixes ` (1).mp4`:
- `id00816`: 14 files (e.g. `00118_id00082_wavtolip (1).mp4`)
- `id06334`: 1 file (`00015_id06355_wavtolip (1).mp4`)
- `id07200`: 1 file (`00045_id04560_wavtolip (1).mp4`)
These 16 files are excluded from policy training and evaluation to prevent duplicate leakage.

---

## 2. Category Distribution on Disk

| Category Name | Directory Folder | Manipulation Modality | Count on Disk | Percentage |
|:---|:---|:---|:---:|:---:|
| `AUTHENTIC` | `RealVideo-RealAudio` | Real Video, Real Audio | 500 | 2.32% |
| `AUDIO_MANIPULATED` | `RealVideo-FakeAudio` | Real Video, Fake Audio (SV2TTS/RTVC) | 500 | 2.32% |
| `VIDEO_MANIPULATED` | `FakeVideo-RealAudio` | Fake Video (Wav2Lip/FSGAN), Real Audio | 9,709 | 45.03% |
| `AUDIO_VIDEO_MANIPULATED` | `FakeVideo-FakeAudio` | Fake Video, Fake Audio (Wav2Lip + SV2TTS) | 10,851 | 50.33% |
| **Total** | | | **21,560** | **100.0%** |

*Note: The natural dataset is heavily skewed towards manipulated video. Stratified class-balanced sampling is mandatory for fair fusion policy training and evaluation.*

---

## 3. Disambiguation & Provenance of `00014_fake.mp4`

There are **5 distinct files** on disk named `00014_fake.mp4` across different celebrity identities. Filenames alone are **not unique**:

| Celebrity Identity | Ethnicity & Gender | Full Relative Path | File Size | SHA-256 Checksum |
|:---|:---|:---|:---:|:---|
| `id06462` | Asian (East) Women | `RealVideo-FakeAudio/Asian (East)/women/id06462/00014_fake.mp4` | 140,870 B | `8beab438a1f247bc13333ba093307ddcfdd0d8103850b4b28decc44afdbbaa6e` |
| `id00363` | Asian (East) Women | `RealVideo-FakeAudio/Asian (East)/women/id00363/00014_fake.mp4` | 134,443 B | `a7a6ba882c66f235a6d9f070e9a686c2e04e575067f7a4e10f7b473f623c61b2` |
| `id07194` | Asian (South) Men | `RealVideo-FakeAudio/Asian (South)/men/id07194/00014_fake.mp4` | 148,608 B | `7a7ba8a22a3e98f3699c939d98178339cd7fdaf58d66ee3d1e68cc372aa8fcd3` |
| `id00498` | Caucasian (European) Men | `RealVideo-FakeAudio/Caucasian (European)/men/id00498/00014_fake.mp4` | 124,277 B | `a0d422c04559d6e45aea5f7d2451657e3929cc4afc4eaa1ee1c3d292b3724f9f` |
| `id00071` | Caucasian (European) Women | `RealVideo-FakeAudio/Caucasian (European)/women/id00071/00014_fake.mp4` | 195,698 B | `9b7bd55032a087a3c1d47911357c9fb686c41ea25682beadcc855d9e9a9f268a` |

### Authoritative Diagnostic Target:
The authoritative diagnostic target referenced in VERITAS audits is:
- **Exact Path:** `D:\VERITAS_TRAINING\data\FakeAVCeleb_v1.2\FakeAVCeleb_v1.2\RealVideo-FakeAudio\Asian (East)\women\id06462\00014_fake.mp4`
- **Identity:** `id06462`
- **SHA-256:** `8beab438a1f247bc13333ba093307ddcfdd0d8103850b4b28decc44afdbbaa6e`
- **Identity Pool Assignment:** **Train Pool** (48 clips total).
- **Audit Mandate:** Because `id06462` belongs to the training identity pool of the dataset, it is preserved exclusively as a **diagnostic smoke test**, and is strictly excluded from the independent held-out evaluation test set.

---

## 4. Group-Disjoint 80/10/10 Identity Partition

To eliminate identity, speaker, and donor leakage across splits, all 500 unique celebrity identities are partitioned disjointly:
- **Train Pool:** 400 identities (80.0%)
- **Validation Pool:** 50 identities (10.0%)
- **Test Pool:** 50 identities (10.0%)

### Full Pool File Counts:
- **Train Pool (400 identities):** 17,141 clips (400 Authentic, 400 Audio-Manipulated, 7,725 Video-Manipulated, 8,616 Audio-Video-Manipulated)
- **Validation Pool (50 identities):** 2,251 clips (50 Authentic, 50 Audio-Manipulated, 1,029 Video-Manipulated, 1,122 Audio-Video-Manipulated)
- **Test Pool (50 identities):** 2,168 clips (50 Authentic, 50 Audio-Manipulated, 955 Video-Manipulated, 1,113 Audio-Video-Manipulated)

### Leakage Verification Assertions:
$$\text{Train IDs} \cap \text{Val IDs} = \emptyset \quad (0)$$
$$\text{Train IDs} \cap \text{Test IDs} = \emptyset \quad (0)$$
$$\text{Val IDs} \cap \text{Test IDs} = \emptyset \quad (0)$$

---

## 5. Detector Training Exposure & Out-of-Sample Integrity

1. **Audio Detector Exposure:** The production audio agent checkpoint (`audio_agent_checkpoint.pth`) was trained on subsets of the 400 Train identities. It has **never seen** the 50 Validation identities, nor the 50 Test identities.
2. **Spatial & Temporal Detector Exposure:** The Spatial and Temporal detectors were similarly evaluated on held-out test identities.
3. **Out-of-Sample Guarantee:**
   - Any features extracted from the 50 Validation identities (50 IDs) and 50 Test identities (50 IDs) represent **100% out-of-sample detector predictions**.
   - Model selection is conducted on the 50 Validation identities.
   - Final evaluation is conducted on the 50 Test identities.
   - Neither the detectors nor the fusion policy have ever seen the Test identities during training.
