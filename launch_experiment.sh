session=whatever
window=${session}:0
pane=${window}.4
tmux send-keys -t "$pane" C-z 'some -new command' Enter
tmux select-pane -t "$pane"
tmux select-window -t "$window"
tmux attach-session -t "$session"
tmux select-window -t:0 'my command'