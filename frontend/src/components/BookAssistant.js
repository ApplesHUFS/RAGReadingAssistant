import { useState, useCallback, useEffect } from 'react';
import { Upload, SendHorizontal } from 'lucide-react';

function BookAssistant() {
  const [selectedBook, setSelectedBook] = useState(null);
  const [books, setBooks] = useState({ available: [], processed: {} });
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [processing, setProcessing] = useState(false);
  const [mode, setMode] = useState('select'); // 'select', 'chat', 'summary'

  const fetchBooks = useCallback(async () => {
    const response = await fetch('http://localhost:8000/api/books');
    const data = await response.json();
    setBooks(data);
  }, []);

  useEffect(() => {
    fetchBooks();
  }, [fetchBooks]);

  useEffect(() => {
    if (selectedBook) {
      askMode();
    }
  }, [selectedBook]);

  const askMode = () => {
    setMessages(prev => [...prev, {
      type: 'assistant',
      content: '작업을 선택해주세요:',
      options: [
        { label: '질문하기', value: 'chat' },
        { label: '요약하기', value: 'summary' },
        { label: '종료하기', value: 'exit' }
      ]
    }]);
    setMode('select');
    setQuery('');
  };

  const handleUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    setLoading(true);
    try {
      await fetch('http://localhost:8000/api/books/upload', {
        method: 'POST',
        body: formData
      });
      await fetchBooks();
    } finally {
      setLoading(false);
    }
  };

  const processBook = async (filename) => {
    setProcessing(true);
    try {
      await fetch(`http://localhost:8000/api/books/process/${filename}`, {
        method: 'POST'
      });
      await fetchBooks();
    } finally {
      setProcessing(false);
    }
  };

  const handleOptionSelect = (value) => {
    if (value === 'exit') {
      setMessages(prev => [...prev,
        { type: 'user', content: '종료하기' },
        { type: 'assistant', content: '대화를 종료합니다. 새로운 대화를 시작하려면 다른 책을 선택해주세요.' }
      ]);
      setMode('select');
      setSelectedBook(null);
      return;
    }

    setMode(value);
    setMessages(prev => [...prev, 
      { type: 'user', content: value === 'chat' ? '질문하기' : '요약하기' },
      { 
        type: 'assistant', 
        content: value === 'chat' ? 
          '어떤 것이 궁금하신가요?' : 
          '요약을 생성하고 있습니다...'
      }
    ]);

    if (value === 'summary') {
      handleSummary();
    }
  };

  const handleChat = async (e) => {
    e.preventDefault();
    if (!selectedBook || !query) return;

    setMessages(prev => [...prev, { type: 'user', content: query }]);
    setQuery('');
    setLoading(true);

    try {
      const response = await fetch(`http://localhost:8000/api/books/${selectedBook}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: query })
      });
      const data = await response.json();
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: data.answer
      }]);
      
      // 답변 후 모드 선택 다시 표시
      setTimeout(askMode, 1000);
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: '죄송합니다. 오류가 발생했습니다.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleSummary = async () => {
    setLoading(true);
    try {
      const response = await fetch(`http://localhost:8000/api/books/${selectedBook}/summary`);
      const data = await response.json();
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: data.final_summary
      }]);
      
      // 요약 후 모드 선택 다시 표시
      setTimeout(askMode, 1000);
    } catch (error) {
      setMessages(prev => [...prev, {
        type: 'assistant',
        content: '요약 생성 중 오류가 발생했습니다.'
      }]);
    } finally {
      setLoading(false);
    }
  };

  const renderMessage = (message) => {
    if (message.options) {
      return (
        <div>
          <p className="mb-3">{message.content}</p>
          <div className="flex gap-2">
            {message.options.map(option => (
              <button
                key={option.value}
                onClick={() => handleOptionSelect(option.value)}
                className={`px-4 py-2 rounded-lg ${
                  option.value === 'exit'
                    ? 'bg-red-100 text-red-600 hover:bg-red-200'
                    : 'bg-blue-100 text-blue-600 hover:bg-blue-200'
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      );
    }

    return <p>{message.content}</p>;
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto p-4">
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <h2 className="text-xl font-bold mb-4">Book Assistant</h2>
          
          <div className="space-y-4">
            <div>
              <h3 className="font-semibold mb-2">사용 가능한 책</h3>
              <div className="space-y-2">
                {books.available.map((book) => (
                  <div key={book.path} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <span>{book.name}</span>
                    <button
                      onClick={() => processBook(book.name)}
                      disabled={processing}
                      className="px-3 py-1 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                    >
                      {processing ? '처리 중...' : '처리하기'}
                    </button>
                  </div>
                ))}

                {books.available.length === 0 && (
                  <p className="text-gray-500 italic">사용 가능한 책이 없습니다.</p>
                )}
              </div>
              
              <div className="mt-4">
                <input
                  type="file"
                  onChange={handleUpload}
                  className="hidden"
                  id="file-upload"
                  accept=".txt"
                />
                <label 
                  htmlFor="file-upload" 
                  className="flex items-center gap-2 text-blue-600 cursor-pointer hover:text-blue-700"
                >
                  <Upload size={20} />
                  새 책 업로드
                </label>
              </div>
            </div>

            <div>
              <h3 className="font-semibold mb-2">처리된 책</h3>
              <select 
                value={selectedBook || ''} 
                onChange={(e) => setSelectedBook(e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option value="">처리된 책을 선택하세요...</option>
                {Object.entries(books.processed).map(([id, meta]) => (
                  <option key={id} value={id}>{meta.file_name}</option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {selectedBook && (
          <div className="bg-white rounded-lg shadow-lg h-[600px] flex flex-col">
            <div className="flex-1 overflow-y-auto p-4 space-y-4">
              {messages.map((message, idx) => (
                <div 
                  key={idx} 
                  className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div 
                    className={`max-w-[80%] rounded-lg p-3 ${
                      message.type === 'user' 
                        ? 'bg-blue-600 text-white' 
                        : 'bg-gray-100'
                    }`}
                  >
                    {renderMessage(message)}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 rounded-lg p-3">
                    <div className="flex gap-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.4s'}}></div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {mode === 'chat' && (
              <form onSubmit={handleChat} className="p-4 border-t">
                <div className="flex gap-2">
                  <input
                    type="text"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="질문을 입력하세요..."
                    className="flex-1 p-2 border rounded-lg"
                  />
                  <button
                    type="submit"
                    disabled={loading || !query}
                    className="bg-blue-600 text-white p-2 rounded-lg disabled:opacity-50"
                  >
                    <SendHorizontal size={20} />
                  </button>
                </div>
              </form>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default BookAssistant;