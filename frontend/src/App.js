import React from 'react';
import { Routes, Route, NavLink } from 'react-router-dom';
import Dashboard from './pages/Dashboard';
import Character from './pages/Character';
import Scraper from './pages/Scraper';
import ViralManager from './pages/ViralManager';
import ContentIdeas from './pages/ContentIdeas';
import Scripts from './pages/Scripts';
import Assets from './pages/Assets';
import Published from './pages/Published';
import MusicManager from './pages/MusicManager';
import Settings from './pages/Settings';
import './App.css';

function App() {
  return (
    <div className="app">
      <nav className="sidebar">
        <div className="logo">
          <h1>Content Pipeline</h1>
        </div>
        <ul className="nav-links">
          <li>
            <NavLink to="/" end>Dashboard</NavLink>
          </li>
          <li>
            <NavLink to="/character">AI Character</NavLink> {/* Added NavLink for AI Character */}
          </li>
          <li>
            <NavLink to="/music">Music Manager</NavLink>
          </li>
          <li>
            <NavLink to="/scraper">Scraper</NavLink>
          </li>
          <li>
            <NavLink to="/viral">Viral Factory</NavLink>
          </li>
          <li>
            <NavLink to="/ideas">Content Ideas</NavLink>
          </li>
          <li>
            <NavLink to="/scripts">Scripts</NavLink>
          </li>
          <li>
            <NavLink to="/assets">Assets</NavLink>
          </li>
          <li>
            <NavLink to="/published">Published</NavLink>
          </li>
          <li>
            <NavLink to="/settings">Settings</NavLink>
          </li>
        </ul>
        <div className="nav-footer">
          <a href="http://100.83.153.43:5678" target="_blank" rel="noopener noreferrer">
            Open n8n
          </a>
        </div>
      </nav>
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/character" element={<Character />} /> {/* Added Route for Character */}
          <Route path="/scraper" element={<Scraper />} />
          <Route path="/viral" element={<ViralManager />} />
          <Route path="/ideas" element={<ContentIdeas />} />
          <Route path="/scripts" element={<Scripts />} />
          <Route path="/assets" element={<Assets />} />
          <Route path="/music" element={<MusicManager />} />
          <Route path="/published" element={<Published />} />
          <Route path="/settings" element={<Settings />} />
        </Routes>
      </main>
    </div>
  );
}

export default App;
