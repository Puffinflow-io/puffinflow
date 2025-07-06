import React from 'react';

interface PuffinLogoProps {
    className?: string;
}

const PuffinLogo: React.FC<PuffinLogoProps> = ({ className }) => {
    return (
        <div className={`flex items-center gap-2 ${className || ''}`} aria-label="Puffinflow Logo">
            <img src="./components/logo.png" alt="" className="h-full w-auto" />
            <span className="text-2xl font-semibold text-current whitespace-nowrap">
                Puffinflow
            </span>
        </div>
    );
};

export default PuffinLogo;
