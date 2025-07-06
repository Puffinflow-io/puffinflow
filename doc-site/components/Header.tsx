
import React from 'react';
import PuffinLogo from './PuffinLogo';

const Header: React.FC = () => {
  const handleNavClick = (e: React.MouseEvent<HTMLAnchorElement>, hash: string) => {
    e.preventDefault();
    window.location.hash = hash;
  };

  return (
    <header className="sticky top-0 bg-[#0A0A0C]/90 z-50 border-b border-white/10">
      <div className="container mx-auto px-6 py-3">
        <div className="flex items-center justify-between">
          <a href="#" onClick={(e) => handleNavClick(e, '')} className="text-gray-50">
            <PuffinLogo className="h-10 w-auto" />
          </a>
          <nav className="hidden md:flex items-center space-x-2">
            <a href="#why-puffinflow" onClick={(e) => handleNavClick(e, '#why-puffinflow')} className="text-gray-300 hover:text-white transition-colors duration-200 font-medium px-3 py-2 rounded-md hover:bg-white/5">Why Puffinflow?</a>
            <a href="#features" onClick={(e) => handleNavClick(e, '#features')} className="text-gray-300 hover:text-white transition-colors duration-200 font-medium px-3 py-2 rounded-md hover:bg-white/5">Features</a>
            <a href="#compare" onClick={(e) => handleNavClick(e, '#compare')} className="text-gray-300 hover:text-white transition-colors duration-200 font-medium px-3 py-2 rounded-md hover:bg-white/5">Tool Comparison</a>
            <a href="#docs" onClick={(e) => handleNavClick(e, '#docs')} className="text-gray-300 hover:text-white transition-colors duration-200 font-medium px-3 py-2 rounded-md hover:bg-white/5">Docs</a>
            <a href="https://github.com/puffinflow-io/puffinflow" target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-white transition-colors duration-200 font-medium px-3 py-2 rounded-md hover:bg-white/5">GitHub</a>
          </nav>
          <div className="flex items-center space-x-4">
            <a
              href="#quickstart"
              onClick={(e) => handleNavClick(e, '#quickstart')}
              className="bg-gradient-to-r from-orange-500 to-orange-600 text-white px-4 py-2 rounded-md hover:from-orange-600 hover:to-orange-700 transition-all duration-300 shadow-lg shadow-orange-600/20 hover:shadow-orange-600/40 font-semibold"
            >
              Get Started
            </a>
          </div>
        </div>
      </div>
    </header>
  );
};

export default React.memo(Header);
