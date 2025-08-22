#  Heart Disease Prediction Dataset – Summary  

This dataset is based on the **2022 Behavioral Risk Factor Surveillance System (BRFSS)** survey. After cleaning and removing missing values, it includes around **400,000 U.S. adults** across all 50 states.

Our aim:  
👉 Predict heart attack risk using personal health data—no invasive tests required.

---

##  What’s Inside?

- **Size:** ~78 MB (too large for GitHub itself)  
- **Records:** ~400,000 individuals  
- **Features:** ~40 columns covering demographics, health habits, wellness, medical history, preventive care, disabilities, and oral health  
- **Target:** `HadHeartAttack` → *Yes* (~6%) / *No* (~94%)

Kaggle Source: [Personal Key Indicators of Heart Disease (Kamil Pytlak)](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease) 

---

##  Demographics  
- `Sex`: Male / Female  
- `AgeCategory`: 18–24 up to 80+  
- `RaceEthnicityCategory`: White, Black, Hispanic, Asian, etc.  
- `State`: U.S. state of residence  

---

##  Health & Wellness  
- `GeneralHealth`: Poor → Excellent  
- `PhysicalHealthDays`, `MentalHealthDays`: Bad health days last month (0–30)  
- `SleepHours`: Avg. sleep per night (2–24 h)  
- `HeightInMeters`, `WeightInKilograms`, `BMI`: Personal measurements  

---

##  Lifestyle & Habits  
- `PhysicalActivities`: Yes / No  
- `SmokerStatus`: Current / Former / Never  
- `AlcoholDrinkers`: Yes / No  
- `ECigaretteUsage`: Current / Former / Never  

---

##  Medical History  
Yes/No indicators for:  
`HadDiabetes`, `HadStroke`, `HadCOPD`, `HadAsthma`, `HadKidneyDisease`,  
`HadArthritis`, `HadDepressiveDisorder`, `HadAngina`, `HadSkinCancer`  

---

##  Preventive Care  
- `FluVaxLast12`, `TetanusLast10Tdap`, `PneumoVaxEver`  
- `HIVTesting`, `ChestScan`, `COVIDPos`, `LastCheckupTime`  

---

##  Disabilities  
Yes/No indicators for:  
`DifficultyWalking`, `DifficultyErrands`, `DifficultyConcentrating`,  
`DifficultyDressingBathing`, `DeafOrHardOfHearing`, `BlindOrVisionDifficulty`  

---

##  Oral Health  
- `RemovedTeeth`: Count of teeth removed due to decay or gum disease  

---

##  Prediction Target  
- **`HadHeartAttack`**:  
  - *Yes*: ~6% of people  
  - *No*: ~94% of people  

> **Note:** The dataset is imbalanced. Successful modeling requires oversampling techniques (e.g., SMOTE) or class-weight adjustments to avoid biases toward the majority class.

---

##  TL;DR  
A rich dataset capturing **health, habits, and demographics** of 400k Americans in 2022—perfect for building reliable, real-world heart attack risk models using **Logistic Regression** (or any classifier).

