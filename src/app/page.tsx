'use client';

import { useState } from 'react';


const Index = () => {
  const [scheduleTime, setScheduleTime] = useState('09:00');

  const handleScheduleChange = (e) => {
    setScheduleTime(e.target.value);
  };

  const handleConfigure = async () => {
    // Implement logic to send configuration to Flask backend
    const response = await fetch('/configure', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ scheduleTime }),
    });

    // Handle response as needed
  };

  const handleRunBot = async () => {
    // Implement logic to trigger bot execution in Flask backend
    const response = await fetch('/run_bot', {
      method: 'POST',
    });

    // Handle response as needed
  };

  return (
    <div>
      <h1>Trading Bot Configuration</h1>
      <label htmlFor="schedule_time">Schedule Time:</label>
      <input
        type="text"
        id="schedule_time"
        name="schedule_time"
        value={scheduleTime}
        onChange={handleScheduleChange}
        required
      />
      <button onClick={handleConfigure}>Save Configuration</button>
      <button onClick={handleRunBot}>Run Bot</button>
    </div>
  );
};

export default Index;
