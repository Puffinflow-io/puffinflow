
import React from 'react';
import PuffinLogo from './PuffinLogo';

const GitHubIcon = () => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className="h-5 w-5"
    viewBox="0 0 20 20"
    fill="currentColor"
  >
    <path
      fillRule="evenodd"
      d="M10 0C4.477 0 0 4.477 0 10c0 4.418 2.865 8.165 6.839 9.49.5.092.682-.217.682-.482 0-.237-.009-.868-.014-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.031-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.378.203 2.398.1 2.651.64.7 1.03 1.595 1.03 2.688 0 3.848-2.338 4.695-4.566 4.942.359.308.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.001 10.001 0 0020 10c0-5.523-4.477-10-10-10z"
      clipRule="evenodd"
    />
  </svg>
);


const Footer: React.FC = () => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <footer className="border-t border-gray-800">
      <div className="container mx-auto px-6 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center text-center md:text-left gap-6">
          <div className="flex flex-col items-center md:items-start text-gray-50">
             <a href="#" onClick={(e) => handleNavClick(e, '')} className="mb-2 inline-block">
              <PuffinLogo className="h-8 w-auto" />
            </a>
            <p className="text-gray-400 text-sm">
              Open Source &bull; MIT Licensed &bull; Built for Production AI
            </p>
          </div>
          <div className="flex flex-col items-center gap-4">
             <a href="https://github.com/puffinflow-io/puffinflow" target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-2 text-sm font-semibold bg-gray-800 hover:bg-gray-700 transition-colors px-4 py-2 rounded-md text-gray-200">
                <GitHubIcon />
                Star us on GitHub
            </a>
          </div>
        </div>
        <div className="mt-8 pt-6 border-t border-gray-800 text-center text-sm text-gray-500">
          <p>&copy; {new Date().getFullYear()} Puffinflow, Inc. All rights reserved.</p>
        </div>
      </div>
    </footer>
  );
};

export default React.memo(Footer);
