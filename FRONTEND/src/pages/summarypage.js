import React, { useState, useEffect, useCallback } from 'react';
import { useLocation } from 'react-router-dom';
import styles from '../styles/summarypage.module.css'; // Import the CSS module

function SummaryPage() {
  const { state } = useLocation();
  const { name, quick_summary, chemical_composition, uses, side_effects } = state.data;

  const pillName = name;
  const quickSummary = quick_summary;
  const composition = chemical_composition;
  const pillUses = uses;
  const sideEffects = side_effects;

  const [selectedLanguage, setSelectedLanguage] = useState("si");
  const [translations, setTranslations] = useState({
    quickSummary: '',
    composition: '',
    pillUses: '',
    sideEffects: '',
  });

  const handleTranslateClick = useCallback(async () => {
    const translatedDescriptions = await fetchTranslatedDescription(selectedLanguage, {
      quickSummary,
      composition,
      pillUses,
      sideEffects,
    });
    setTranslations(translatedDescriptions);
  }, [selectedLanguage, quickSummary, composition, pillUses, sideEffects]);

  useEffect(() => {
    handleTranslateClick();
  }, [selectedLanguage, handleTranslateClick]); // Trigger translation when language changes

  const handleLanguageChange = (e) => {
    setSelectedLanguage(e.target.value);
  };

  const fetchTranslatedDescription = async (language, text) => {
    try {
      const response = await fetch('https://medscan-backend-t7mjv53jma-uc.a.run.app/translate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add your API key here
          'Authorization': '9b970bd18d484ee2860ad4c62ae44681'
        },
        body: JSON.stringify({
          text: {
            quickSummary: text.quickSummary,
            composition: text.composition,
            pillUses: text.pillUses,
            sideEffects: text.sideEffects,
          },
          to: language
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch translated description');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error fetching translated description:', error);
      return '';
    }
  };
  
  return (
    <div className={styles['background-image']}>
           
      <div className={styles['summary-container']}>
      <h1 className={styles.heading}>Here's what we found:</h1>

        <div className={styles.medicineTitle}>Medicine Name:</div>
        <div className={styles.medicineName}>'{pillName}'</div>
        <div className={styles['description-container']}>
          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Quick Summary:</h1>
            <div className={styles.detailsParaContainer}>
            <p className={styles.detailsPara}>{quickSummary}</p>
            <p className={styles.detailsPara}>{translations.quickSummary}</p>
            </div>
          </div>
          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Chemical Composition:</h1>
            <div className={styles.detailsParaContainer}>
            <p className={styles.detailsPara}>{composition}</p>
            <p className={styles.detailsPara}>{translations.composition}</p>
            </div>
          </div>
          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Pill Benefits:</h1>
            <div className={styles.detailsParaContainer}>
            <p className={styles.detailsPara}>{pillUses}</p>
            <p className={styles.detailsPara}>{translations.pillUses}</p>
            </div>
          </div>
          <div className={styles.detailsOverview}>
            <h1 className={styles.detailsHeading}>Side Effects:</h1>
            <div className={styles.detailsParaView}>
            <p className={styles.detailsPara}>{sideEffects}</p>
            <p className={styles.detailsPara}>{translations.sideEffects}</p>
            </div>
          </div>
          <div className={styles['dropdown-container']}>
            <select className={styles['buttonDropdown']} value={selectedLanguage} onChange={handleLanguageChange}>
              <option value="si">Sinhala</option>
              <option value="ta">Tamil</option>
            </select>
            <button className={styles['buttonDropdown']} onClick={handleTranslateClick}>Translate</button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SummaryPage;


