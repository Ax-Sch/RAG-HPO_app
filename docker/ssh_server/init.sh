#!/bin/bash

# Parse SSH_USERS environment variable (format: user1:password1,user2:password2)
IFS=',' read -r -a users <<< "$(</users.txt)"

echo "PasswordAuthentication yes
        Match User *
        ForceCommand sleep infinity
        AllowTcpForwarding yes
        PermitTTY no"  >> /etc/ssh/sshd_config


for user in "${users[@]}"; do
  username=$(echo $user | cut -d':' -f1)
  password=$(echo $user | cut -d':' -f2)

  # Check if user exists, if not, create the user
  if id "$username" &>/dev/null; then
    echo "User $username already exists."
  else
    # Create user with no shell access by default
    useradd -m -s "/bin/bash" $username
    echo "$username:$password" | chpasswd
    echo "User $username created."
  fi

  # Set up SSH directory and permissions
  mkdir -p /home/$username/.ssh
  chown -R $username:$username /home/$username/.ssh
  chmod 700 /home/$username/.ssh

done

# Restart SSH service to apply changes
service ssh restart

# Keep the container running
echo "container initiated."