import NavigationBar from "../components/navigationBar";
import Footer from "../components/footer";
import React from 'react';
import { BrowserRouter, Routes, Route, } from 'react-router-dom';
import Homepage from "../pages/homepage";
import SearchPage from "../pages/searchpage";
import SummaryPage from "../pages/summarypage";


const Routers = () => {
  return (
    <BrowserRouter>
      <NavigationBar />
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route path="/search-page" element={<SearchPage />} />
        <Route path="/summary-page" element={<SummaryPage />} />
      </Routes>
      <Footer />
    </BrowserRouter>
  );
}

export default Routers;