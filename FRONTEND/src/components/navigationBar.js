import { useCallback } from "react";
import { useNavigate } from "react-router-dom";
import styles from "./navigationbar.module.css";
import medScanLogo from "../images/icons/medscan.png";


const NavigationBar = () => {
  const navigate = useNavigate();

  const onHomeTextClick = useCallback(() => {
    navigate("/");
  }, [navigate]);

  const onSearchTextClick = useCallback(() => {
    navigate("/search-page");
  }, [navigate]);

  return (
    <div className={styles.navigationBar}>
        <div className={styles.logo} onClick={onHomeTextClick}>
          <img className={styles.medscanIcon} alt="logo" src={medScanLogo} />
          <b className={styles.medscanTxt}>OPTIDETECT</b>
        </div>
          <div className={styles.pages}>
            <div className={styles.pages} onClick={onHomeTextClick}> Home </div>
            <div className={styles.pages} onClick={onSearchTextClick}> Search </div>
          </div>
    </div>
  );
};

export default NavigationBar;
