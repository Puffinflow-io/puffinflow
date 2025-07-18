import React, { useState, useEffect, useRef } from 'react';
import { ClipboardIcon, ClipboardCheckIcon } from './Icons';

declare const Prism: any;

interface CodeWindowProps {
  code: string;
  language: string;
  fileName?: string;
  showLineNumbers?: boolean;
}

const CodeWindow: React.FC<CodeWindowProps> = ({ code, language, fileName, showLineNumbers = true }) => {
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

         <div className="code-container relative">
           <div className="flex">
             {showLineNumbers && (
               <div className="line-numbers-container select-none text-gray-500 text-right pr-4 border-r border-gray-700 bg-[#1a1d24] font-mono text-sm leading-6">
                 {code.split('\n').map((_, index) => (
                   <div key={index} className="line-number" style={{ minWidth: `${Math.max(2, String(code.split('\n').length).length)}ch` }}>
                     {index + 1}
                   </div>
                 ))}
               </div>
             )}
             <pre className={`language-${language} !m-0 !rounded-none flex-1 !pl-4 !bg-transparent`}>
               <code ref={codeRef} className={`language-${language}`}>
                  {/* Populated by useEffect */}
               </code>
             </pre>
           </div>
         </div>
     </div>
  );
};

export default CodeWindow;
