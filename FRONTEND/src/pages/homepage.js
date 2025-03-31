import React, { useState, useEffect, useRef } from 'react';
import styles from "../styles/homepage.module.css";
import { useNavigate } from 'react-router-dom';
import pills from "../images/icons/pills.png";

const HomePage = () => {
  const navigate = useNavigate();
  const [isVisible, setIsVisible] = useState(false);
  const [isNavBoxSetVisible, setIsNavBoxSetVisible] = useState(false);
  const hiddenElementRef = useRef(null);
  const navBoxSetRef = useRef(null);

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        setIsVisible(entry.isIntersecting);
      });
    });

    const hiddenElement = hiddenElementRef.current;
    if (hiddenElement) {
      observer.observe(hiddenElement);
    }

    const navBoxSetObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        setIsNavBoxSetVisible(entry.isIntersecting);
      });
    });

    const navBoxSetElement = navBoxSetRef.current;
    if (navBoxSetElement) {
      navBoxSetObserver.observe(navBoxSetElement);
    }

    return () => {
      if (hiddenElement) {
        observer.unobserve(hiddenElement);
      }
      if (navBoxSetElement) {
        navBoxSetObserver.unobserve(navBoxSetElement);
      }
    };
  }, []);

  const handleSearch = () => {
    navigate('/search-page');
  };

  return (
    <div className={styles.homepage}>
      <div className={styles.topset}>
        <div ref={hiddenElementRef} className={`${styles.hidden} ${isVisible ? styles.show : ''} ${styles.topsetleft}`}>
          <h4 className={styles.welcome}> Welcome to </h4>
          <h1 className={styles.medscan}> OPTIDETECT </h1>
          <p className={styles.startinfo}> We provide eye information to your fingertips </p>
          <button onClick={handleSearch} className={styles.searchPageButton}>
            <div className={styles.button}>Get Information</div>
          </button>
        </div>
        
        <div className={styles.topsetright}>
            <img className={styles.pillsImg} alt="logo" src={pills} />
        </div>
      </div>
    </div>
  );
};

export default HomePage;
