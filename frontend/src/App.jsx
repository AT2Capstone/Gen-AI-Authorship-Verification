import React from 'react'
import Detector from './components/Detector'

export default function App() {
  return (
    <div className="container my-4">
      <header className="mb-4">
        <h1 className="h3">AI Content Detector</h1>
        <p className="text-muted"></p>
      </header>

      <Detector />
      <footer className="mt-4 text-muted small">Built with FastAPI</footer>
    </div>
  )
}