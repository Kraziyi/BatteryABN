import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import ProjectList from './components/ProjectList';
import CellList from './components/CellList';
import CellDetails from './components/CellDetails';
import CellSearch from './components/CellSearch';
import TestRecordList from './components/TestRecordList';
import TestRecordDetails from './components/TestRecordDetails';
import TestRecordSearch from './components/TestRecordSearch';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <main>
          <Routes>
            <Route path="/" element={<ProjectList />} />
            <Route path="/cells" element={<CellSearch />} />
            <Route path="/cells/:projectName" element={<CellList />} />
            <Route path="/cell/:cellName" element={<CellDetails />} />
            <Route path="/trs" element={<TestRecordSearch />} />
            <Route path="/trs/:cellName" element={<TestRecordList />} />
            <Route path="/tr/:trName" element={<TestRecordDetails />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;