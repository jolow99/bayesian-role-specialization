# Bayesian Role Specialization Experiment

This is an Empirica implementation of a cooperative battle game designed to study how humans infer and coordinate their roles in teams through Bayesian inference. Three players work together to defeat an enemy by selecting their role at each round. Players must coordinate without explicit communication by inferring each other's roles from observed actions.

## Setup and Running Locally

1. Install Empirica:
```bash
curl -fsS https://install.empirica.dev | sh
```

2. Clone this repository:
```bash
git clone https://github.com/yourusername/bayesian-role-specialization.git
cd bayesian-role-specialization
```

3. Start the Empirica server:
```bash
empirica
```

This will start the server on `http://localhost:3000` and automatically build and watch the client and server code.

4. Open `http://localhost:3000/admin` in your browser to access the admin panel where you can create batches and configure treatments.

5. Open `http://localhost:3000` in 3 separate browser windows/tabs to test with 3 players.

## Deployment

### Initial Setup

1. **Create a DigitalOcean Droplet**
   - Ubuntu 22.04 or newer
   - Note the IP address (e.g., `152.24.201.32`)

2. **Configure DNS**
   - Add an A record for your subdomain pointing to the droplet IP
   - Example: `experiment.domain.com` → `INSERT_IP_ADDRESS`
   - Turn off Cloudflare proxy (orange cloud) if using Cloudflare

3. **SSH into the server**
   ```bash
   ssh root@INSERT_IP_ADDRESS
   ```

4. **Install Empirica**
   ```bash
   curl https://install.empirica.dev | sh -s
   ```

5. **Install Caddy (reverse proxy with automatic HTTPS)**
   ```bash
   sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https curl
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
   chmod o+r /usr/share/keyrings/caddy-stable-archive-keyring.gpg
   chmod o+r /etc/apt/sources.list.d/caddy-stable.list
   sudo apt update
   sudo apt install caddy
   ```

6. **Configure Caddy**
   ```bash
   sudo nano /etc/caddy/Caddyfile
   ```

   Replace the entire contents with:
   ```
   experiment.domain.com {
       reverse_proxy localhost:3000
   }
   ```

   Reload Caddy:
   ```bash
   sudo systemctl reload caddy
   ```

7. **Deploy the experiment (first time)**

   On your local machine:
   ```bash
   empirica bundle
   scp bayesian-role-specialization.tar.zst root@INSERT_IP_ADDRESS:~/
   ```

   On the server:
   ```bash
   screen -S empirica
   empirica serve bayesian-role-specialization.tar.zst
   # Press Ctrl+A then D to detach from screen
   ```

8. **Access your experiment**
   - Visit `https://experiment.domain.com`
   - Caddy automatically handles HTTPS certificates via Let's Encrypt

### Updating the Deployment

When you make changes to your experiment:

1. **Bundle the updated experiment** (on your local machine):
   ```bash
   empirica bundle
   ```

2. **Copy the new bundle to the server**:
   ```bash
   scp bayesian-role-specialization.tar.zst root@INSERT_IP_ADDRESS:~/
   ```

3. **Restart Empirica on the server**:
   ```bash
   ssh root@INSERT_IP_ADDRESS
   screen -r empirica  # Reattach to the screen session
   # Press Ctrl+C to stop the current process
   empirica serve bayesian-role-specialization.tar.zst
   # Press Ctrl+A then D to detach
   ```

### Useful Commands

- **Check if Caddy is running**: `systemctl status caddy`
- **View Caddy logs**: `journalctl -u caddy --no-pager | less +G`
- **Reattach to Empirica screen**: `screen -r empirica`
- **List all screen sessions**: `screen -ls`
- **Check Empirica is running**: Look for process with `ps aux | grep empirica`

### Backing Up Data

The experiment data is stored in `/root/.empirica/local/tajriba.json` on the server. To back it up:

```bash
scp root@INSERT_IP_ADDRESS:~/.empirica/local/tajriba.json ./backup-$(date +%Y%m%d).json
``` 
