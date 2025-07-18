import React, { useState } from 'react';

interface CodeBlockProps {
  code: string;
  language: string;
}

const CodeBlock: React.FC<CodeBlockProps> = ({ code, language }) => {
  const [copyText, setCopyText] = useState('Copy');

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopyText('Copied!');
      setTimeout(() => setCopyText('Copy'), 2000);
    });
  };

  return (
    <div className="code-block text-left">
      <button onClick={handleCopy} className="copy-button">
        {copyText}
      </button>
      <div className="code-container relative">
        <div className="flex">
          <div className="line-numbers-container select-none text-gray-500 text-right pr-4 border-r border-gray-700 bg-[#1a1d24] font-mono text-sm leading-6">
            {code.split('\n').map((_, index) => (
              <div key={index} className="line-number" style={{ minWidth: `${Math.max(2, String(code.split('\n').length).length)}ch` }}>
                {index + 1}
              </div>
            ))}
          </div>
          <pre className={`language-${language} !m-0 !rounded-none flex-1 !pl-4 !bg-transparent`}>
            <code className={`language-${language}`}>{code}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};

export default CodeBlock;
