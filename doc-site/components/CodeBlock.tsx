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
      <pre>
        <code className={`language-${language}`}>{code}</code>
      </pre>
    </div>
  );
};

export default CodeBlock;
