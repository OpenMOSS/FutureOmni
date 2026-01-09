import json
import os
from collections import defaultdict
import pdb
import re
import datetime
def sort_by_time_string(strings):
    """
    Sort a list of strings by embedded timestamp in format YYYYMMDD_HHMMSS
    
    Args:
        strings: List of strings containing timestamps like "20250912_105046"
    
    Returns:
        List of strings sorted by timestamp (oldest to newest)
    """
    def extract_timestamp(s):
        # Pattern to match YYYYMMDD_HHMMSS format
        pattern = r'(\d{8}_\d{6})'
        match = re.search(pattern, s)
        if match:
            timestamp_str = match.group(1)
            # Convert to datetime object for proper sorting
            return datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
        else:
            # Return min datetime if no timestamp found (puts items without timestamps first)
            return datetime.min
    
    return sorted(strings, key=extract_timestamp)

def trans_seconds2(timestamp):
    # "13:51-14:00"
    width = len(timestamp) // 2
    start = timestamp[:width]
    end = timestamp[width+1:]
    
    def _trans(time):
        time = time.strip(":")
        if '-' in time:
            time = time.split("-")[0]
            
        time_chunks = time.split(":")
        
        # Convert to seconds based on number of components
        # HH:MM:SS, MM:SS, or SS
        if len(time_chunks) == 3:
            seconds = int(time_chunks[0]) * 3600 + int(time_chunks[1]) * 60 + int(time_chunks[2])
        elif len(time_chunks) == 2:
            seconds = int(time_chunks[0]) * 60 + int(time_chunks[1])
        elif len(time_chunks) == 1:
            seconds = int(time_chunks[0])
        else:
            seconds = 0
            
        return seconds
    
    start_seconds = _trans(start)
    end_seconds = _trans(end)
    return start_seconds, end_seconds


        

    