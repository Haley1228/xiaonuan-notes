---
data: 2026-05-08
---
# 1.Experimental Settings
## 1.Datasets.

>We assess the performance of our method on multimodal sentiment analysis benchmark datasets, including IEMOCAP (Busso et al. 2008) and MELD (Poria et al. 2018) and CMU-MOSEI (Zadeh et al. 2018) datasets. The statistics are reported in Table 1. Both of them are multimodal datasets with textual, visual, and acoustic modalities
>
>CMU-MOSI [40] is one of the most widely used benchmark dataset in the field of MSA. It contains 2199 utterance video segments, and each segment is manually annotated with a sentiment score ranging from -3 to +3 to indicate the sentiment polarity and relative sentiment strength of the segment. CMU-MOSEI [1] is an upgraded version of CMU-MOSI. It comprises 23,453 video segments from 3,228 videos, similar to MOSI, each utterance-level sample in MOSEI is annotated a sentiment label on a scale of -3 to 3.
>
>We evaluate PRISM on three widely used multimodal sentiment analysis benchmarks, whose statistics are summarized in Table 1. CMU-MOSI [59] and CMU-MOSEI [60] are English video sentiment datasets, while CH-SIMS [55] is a Chinese dataset. CMU-MOSI  provides a relatively small benchmark, CMU-MOSEI offers a larger and more diverse testbed, and CH-SIMS extends evaluation to a different language and label granularity.
>,
>We conduct extensive experiments on two mainstream datasets, including CMU-MOSI [69] and CMU-MOSEI [70]. MOSI. 
>The MOSI dataset contains 2,199 aligned monologue video clips (about 18 hours total) with tri-modal features extracted at 12.5 Hz (audio) and 15 Hz (visual). Following standard partitions, the corpus splits into 1,284 training, 229 validation, and 686 test samples. Each clip is annotated with fine-grained sentiment intensity scores on a 7-point scale (-3: strongly negative to +3: strongly positive). 
>MOSEI. As the expanded successor to MOSI, the MOSEI dataset scales to 22,856 video samples (about 65 hours) featuring enhanced feature extraction at 20 Hz (audio) and 15 Hz (visual). The dataset adopts a standardized split of 16,326 training, 1,871 validation, and 4,659 test samples. It retains MOSI’s continuous sentiment scoring (-3 to +3) while introducing additional sentiment categories, though sentiment intensity remains the primary annotation for consistency.
>
>The experiments were conducted on three publicly available benchmark datasets in MSA: CMU-MOSI [Zadeh et al., 2016], CMU-MOSEI [Zadeh et al., 2018] and CHSIMS [Yu et al., 2020]