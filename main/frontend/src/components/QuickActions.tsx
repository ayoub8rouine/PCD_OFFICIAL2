import React from 'react';

interface QuickActionsProps {
  onSelectAction: (action: string) => void;
  disabled: boolean;
}

const QuickActions: React.FC<QuickActionsProps> = ({ onSelectAction, disabled }) => {
  const quickActions = [
    "Common cold symptoms",
    "Headache remedies",
    "COVID-19 information",
    "First aid for burns",
    "Medication interactions",
    "Blood pressure concerns"
  ];

  return (
    <div className="p-4 bg-white border-t border-gray-200">
      <div className="container mx-auto max-w-3xl">
        <p className="text-sm text-gray-500 mb-2">Quick questions:</p>
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={() => !disabled && onSelectAction(action)}
              className={`text-sm px-3 py-1.5 rounded-full border ${
                disabled
                  ? 'border-gray-200 text-gray-400 cursor-not-allowed'
                  : 'border-blue-200 text-blue-600 hover:bg-blue-50'
              } transition-colors`}
              disabled={disabled}
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};

export default QuickActions;