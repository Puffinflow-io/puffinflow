import React, { useState, useEffect, useRef } from 'react';
import { ClipboardIcon, ClipboardCheckIcon } from './Icons';

declare const Prism: any;

interface CodeWindowProps {
  code: string;
  language: string;
  fileName?: string;
}

const CodeWindow: React.FC<CodeWindowProps> = ({ code, language, fileName }) => {
  const [copyText, setCopyText] = useState('Copy');
  const codeRef = useRef<HTMLElement>(null);

  useEffect(() => {
    if (codeRef.current && typeof Prism !== 'undefined') {
        codeRef.current.textContent = code;
        Prism.highlightElement(codeRef.current);
    }
  }, [code, language]);

  const handleCopy = () => {
    navigator.clipboard.writeText(code).then(() => {
      setCopyText('Copied!');
      setTimeout(() => setCopyText('Copy'), 2000);
    });
  };

  return (
     <div className="code-window-wrapper my-6 not-prose bg-[#16181D] rounded-xl shadow-2xl overflow-hidden border border-white/10">
         {/* Window Header */}
         <div className="flex items-center justify-between px-4 py-2 bg-[#22252A] border-b border-black/20">
             <div className="flex items-center space-x-2">
                 <div className="w-3 h-3 bg-[#ff5f56] rounded-full border border-black/30"></div>
                 <div className="w-3 h-3 bg-[#ffbd2e] rounded-full border border-black/30"></div>
                 <div className="w-3 h-3 bg-[#27c93f] rounded-full border border-black/30"></div>
             </div>
             {fileName && <p className="text-sm text-gray-400 font-mono select-none">{fileName}</p>}
             <button onClick={handleCopy} className="flex items-center gap-1.5 text-xs text-gray-400 hover:text-white transition-colors duration-200">
               {copyText === 'Copied!' ? (
                 <ClipboardCheckIcon className="h-4 w-4 text-green-400" />
               ) : (
                 <ClipboardIcon className="h-4 w-4" />
               )}
               {copyText}
             </button>
         </div>

         <pre className={`line-numbers language-${language} !m-0 !rounded-none`}>
             <code ref={codeRef} className={`language-${language}`}>
                {/* Populated by useEffect */}
             </code>
         </pre>
     </div>
  );
};

export default CodeWindow;
