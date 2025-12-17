import React, { useState, useEffect } from 'react'
import './App.css'

const API_URL = 'http://localhost:8000'

function App() {
  const [tools, setTools] = useState([])
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Hello! Ask me anything about LangChain, ChromaDB, or other documentation.'
    }
  ])
  const [query, setQuery] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  useEffect(() => {
    loadTools()
  }, [])

  const loadTools = async () => {
    try {
      const response = await fetch(`${API_URL}/tools`)
      const data = await response.json()
      setTools(data.tools || [])
    } catch (err) {
      console.error('Failed to load tools:', err)
    }
  }

  const sendQuery = async () => {
    if (!query.trim() || loading) return

    const userMessage = { role: 'user', content: query }
    setMessages(prev => [...prev, userMessage])
    setQuery('')
    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`${API_URL}/query`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query }),
      })

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }))
        throw new Error(`API error: ${response.status} - ${errorData.detail || errorData.message || 'Unknown error'}`)
      }

      const data = await response.json()
      setMessages(data.messages || [])
    } catch (err) {
      setError(err.message)
      console.error('Query error:', err)
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendQuery()
    }
  }

  const renderMessageContent = (content) => {
    if (typeof content === 'string') {
      return <div className="message-text">{content}</div>
    }
    
    if (Array.isArray(content)) {
      return content.map((item, idx) => {
        if (item.type === 'tool_use') {
          return (
            <div key={idx} className="tool-call">
              🔧 Called tool: <strong>{item.name}</strong>
              <pre>{JSON.stringify(item.input, null, 2)}</pre>
            </div>
          )
        }
        if (item.type === 'tool_result') {
          return (
            <div key={idx} className="tool-result">
              ✅ Tool result:
              <pre>{JSON.stringify(item.content, null, 2)}</pre>
            </div>
          )
        }
        return null
      })
    }
    
    return <div className="message-text">{JSON.stringify(content)}</div>
  }

  return (
    <div className="container">
      <div className="header">
        <h1>MCP Client</h1>
      </div>

      <div className="content-wrapper">
        <div className="sidebar">
        <div className="sidebar-section">
          <h2>Settings</h2>
          <div className="info">
            API: <span>{API_URL}</span>
          </div>
        </div>

        <div className="sidebar-section">
          <h2>Available Tools</h2>
          {tools.length === 0 ? (
            <div className="loading">Loading tools...</div>
          ) : (
            <ul className="tools-list">
              {tools.map((tool, idx) => (
                <li key={idx}>{tool.name}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      <div className="chat-area">
        <div className="messages">
          {messages.map((msg, idx) => (
            <div key={idx} className={`message ${msg.role}`}>
              <div className="message-role">{msg.role}</div>
              {renderMessageContent(msg.content)}
            </div>
          ))}
          {loading && (
            <div className="message assistant">
              <div className="loading">Thinking...</div>
            </div>
          )}
        </div>

        {error && (
          <div className="error">
            Error: {error}. Make sure the API is running on {API_URL}
          </div>
        )}

        <div className="input-area">
          <input
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Enter your query here..."
            disabled={loading}
          />
          <button onClick={sendQuery} disabled={loading || !query.trim()}>
            {loading ? 'Sending...' : 'Send'}
          </button>
        </div>
        </div>
      </div>
    </div>
  )
}

export default App

