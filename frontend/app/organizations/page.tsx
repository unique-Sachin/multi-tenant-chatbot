'use client';

import { useState, useEffect } from 'react';
import { ProtectedRoute } from '@/components/auth/protected-route';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import {
  Building2,
  Globe,
  PlayCircle,
  Plus,
  RefreshCw,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
} from 'lucide-react';
import { organizationService } from '@/lib/services/organizations';
import type { Organization, Website, IngestionJob } from '@/lib/types';
import { toast } from 'sonner';
import Link from 'next/link';

export default function OrganizationsPage() {
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [websites, setWebsites] = useState<Website[]>([]);
  const [jobs, setJobs] = useState<IngestionJob[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isLoadingJobs, setIsLoadingJobs] = useState(false);
  const [selectedOrgId, setSelectedOrgId] = useState<string>('all');
  const [selectedOrgIdForWebsite, setSelectedOrgIdForWebsite] = useState<string>('');
  const [activeTab, setActiveTab] = useState('organizations');

  useEffect(() => {
    loadOrganizationsAndWebsites();
  }, []);

  // Load jobs only when the jobs tab is active
  useEffect(() => {
    if (activeTab === 'jobs' && websites.length > 0) {
      loadAllJobs();
    }
  }, [activeTab, websites]);

  const loadOrganizationsAndWebsites = async () => {
    setIsLoading(true);
    try {
      const [orgsData, websitesData] = await Promise.all([
        organizationService.getOrganizations(),
        organizationService.getWebsites(),
      ]);
      setOrganizations(orgsData);
      setWebsites(websitesData);
    } catch (error) {
      toast.error('Failed to load data');
    } finally {
      setIsLoading(false);
    }
  };

  const loadAllJobs = async () => {
    setIsLoadingJobs(true);
    try {
      // Fetch jobs for all websites
      const jobPromises = websites.map(website => 
        organizationService.getIngestionJobs(website.id).catch(() => [])
      );
      const jobArrays = await Promise.all(jobPromises);
      const allJobs = jobArrays.flat();
      setJobs(allJobs);
    } catch (error) {
      toast.error('Failed to load ingestion jobs');
    } finally {
      setIsLoadingJobs(false);
    }
  };

  const handleCreateOrganization = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const name = formData.get('name') as string;
    const description = formData.get('description') as string;

    try {
      await organizationService.createOrganization({ name, description });
      toast.success('Organization created successfully');
      await loadOrganizationsAndWebsites();
      (e.target as HTMLFormElement).reset();
    } catch (error) {
      toast.error('Failed to create organization');
    }
  };

  const handleAddWebsite = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const url = formData.get('url') as string;

    if (!selectedOrgIdForWebsite) {
      toast.error('Please select an organization');
      return;
    }

    try {
      await organizationService.createWebsite({ org_id: selectedOrgIdForWebsite, url });
      toast.success('Website added successfully');
      await loadOrganizationsAndWebsites();
      setSelectedOrgIdForWebsite('');
      (e.target as HTMLFormElement).reset();
    } catch (error) {
      toast.error('Failed to add website');
    }
  };

  const handleStartIngestion = async (websiteId: string) => {
    try {
      await organizationService.createIngestionJob(websiteId);
      toast.success('Ingestion job started');
      // Reload websites to update status
      await loadOrganizationsAndWebsites();
      // If on jobs tab, reload jobs
      if (activeTab === 'jobs') {
        await loadAllJobs();
      }
    } catch (error) {
      toast.error('Failed to start ingestion');
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle2 className="h-4 w-4 text-green-500" />;
      case 'failed':
        return <XCircle className="h-4 w-4 text-red-500" />;
      case 'running':
      case 'ingesting':
        return <Loader2 className="h-4 w-4 text-blue-500 animate-spin" />;
      default:
        return <Clock className="h-4 w-4 text-yellow-500" />;
    }
  };

  const filteredWebsites = selectedOrgId && selectedOrgId !== 'all'
    ? websites.filter((w) => w.org_id === selectedOrgId)
    : websites;

  return (
    <ProtectedRoute>
      <div className="min-h-screen bg-background">
        {/* Header */}
        <div className="border-b">
          <div className="container mx-auto px-4 py-4">
            <div className="flex items-center justify-between">
              <div>
                <h1 className="text-2xl font-bold">Organization Management</h1>
                <p className="text-muted-foreground">
                  Manage organizations, websites, and ingestion jobs
                </p>
              </div>
              <div className="flex gap-2">
                <Link href="/chat">
                  <Button variant="outline">Back to Chat</Button>
                </Link>
                <Button onClick={loadOrganizationsAndWebsites} disabled={isLoading}>
                  <RefreshCw className={`mr-2 h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
                  Refresh
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="container mx-auto px-4 py-8">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
            <TabsList>
              <TabsTrigger value="organizations">Organizations</TabsTrigger>
              <TabsTrigger value="websites">Websites</TabsTrigger>
              <TabsTrigger value="jobs">Ingestion Jobs</TabsTrigger>
            </TabsList>

            <TabsContent value="organizations" className="space-y-4">
              {/* Create Organization Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Plus className="h-5 w-5" />
                    Create Organization
                  </CardTitle>
                  <CardDescription>
                    Add a new organization to manage its websites and knowledge base
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleCreateOrganization} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="org-name">Organization Name</Label>
                      <Input
                        id="org-name"
                        name="name"
                        placeholder="e.g., Acme Corp"
                        required
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="org-description">Description (optional)</Label>
                      <Textarea
                        id="org-description"
                        name="description"
                        placeholder="Describe this organization"
                        rows={3}
                      />
                    </div>
                    <Button type="submit">Create Organization</Button>
                  </form>
                </CardContent>
              </Card>

              {/* Organizations List */}
              <div className="grid gap-4">
                {organizations.length === 0 ? (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Building2 className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">No organizations yet</p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Create your first organization above
                      </p>
                    </CardContent>
                  </Card>
                ) : (
                  organizations.map((org) => (
                    <Card key={org.id}>
                    <CardHeader>
                      <div className="flex items-start justify-between">
                        <div>
                          <CardTitle className="flex items-center gap-2">
                            <Building2 className="h-5 w-5" />
                            {org.name}
                          </CardTitle>
                          <CardDescription className="mt-1">
                            <span className="font-mono text-xs">{org.slug}</span>
                          </CardDescription>
                        </div>
                        <Badge variant="outline">
                          {websites.filter((w) => w.org_id === org.id).length} websites
                        </Badge>
                      </div>
                    </CardHeader>
                    {org.description && (
                      <CardContent>
                        <p className="text-sm text-muted-foreground">{org.description}</p>
                      </CardContent>
                    )}
                  </Card>
                ))
                )}
              </div>
            </TabsContent>

            <TabsContent value="websites" className="space-y-4">
              {/* Add Website Card */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Plus className="h-5 w-5" />
                    Add Website
                  </CardTitle>
                  <CardDescription>
                    Add a website to an organization for content ingestion
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <form onSubmit={handleAddWebsite} className="space-y-4">
                    <div className="space-y-2">
                      <Label htmlFor="website-org">Organization</Label>
                      <Select 
                        value={selectedOrgIdForWebsite} 
                        onValueChange={setSelectedOrgIdForWebsite}
                        required
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select organization" />
                        </SelectTrigger>
                        <SelectContent>
                          {organizations.map((org) => (
                            <SelectItem key={org.id} value={org.id}>
                              {org.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="website-url">Website URL</Label>
                      <Input
                        id="website-url"
                        name="url"
                        type="url"
                        placeholder="https://example.com"
                        required
                      />
                    </div>
                    <Button type="submit">Add Website</Button>
                  </form>
                </CardContent>
              </Card>

              {/* Filter */}
              <div className="flex gap-2">
                <Select value={selectedOrgId} onValueChange={setSelectedOrgId}>
                  <SelectTrigger className="w-[250px]">
                    <SelectValue placeholder="All Organizations" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">All Organizations</SelectItem>
                    {organizations.map((org) => (
                      <SelectItem key={org.id} value={org.id}>
                        {org.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Websites List */}
              <div className="grid gap-4">
                {filteredWebsites.length === 0 ? (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <Globe className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-muted-foreground">
                        {selectedOrgId && selectedOrgId !== 'all' ? 'No websites for this organization' : 'No websites yet'}
                      </p>
                      <p className="text-sm text-muted-foreground mt-2">
                        Add a website above to get started
                      </p>
                    </CardContent>
                  </Card>
                ) : (
                  filteredWebsites.map((website) => {
                  const org = organizations.find((o) => o.id === website.org_id);
                  return (
                    <Card key={website.id}>
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <CardTitle className="flex items-center gap-2 text-base">
                              <Globe className="h-4 w-4" />
                              {website.url}
                            </CardTitle>
                            <CardDescription className="mt-1">
                              Organization: {org?.name || 'Unknown'}
                              <span className="ml-2">•</span>
                              <span className="ml-2 font-mono text-xs">
                                {website.namespace}
                              </span>
                            </CardDescription>
                          </div>
                          <div className="flex items-center gap-2">
                            <Badge variant={website.status === 'completed' ? 'default' : 'secondary'} className="gap-1">
                              {getStatusIcon(website.status)}
                              {website.status}
                            </Badge>
                            {website.status !== 'ingesting' && (
                              <Button
                                size="sm"
                                onClick={() => handleStartIngestion(website.id)}
                              >
                                <PlayCircle className="mr-1 h-4 w-4" />
                                Ingest
                              </Button>
                            )}
                          </div>
                        </div>
                      </CardHeader>
                      {(website.pages_crawled || website.chunks_created) && (
                        <CardContent>
                          <div className="flex gap-6 text-sm">
                            {website.pages_crawled && (
                              <div>
                                <span className="text-muted-foreground">Pages: </span>
                                <span className="font-medium">{website.pages_crawled}</span>
                              </div>
                            )}
                            {website.chunks_created && (
                              <div>
                                <span className="text-muted-foreground">Chunks: </span>
                                <span className="font-medium">{website.chunks_created}</span>
                              </div>
                            )}
                          </div>
                        </CardContent>
                      )}
                    </Card>
                  );
                })
                )}
              </div>
            </TabsContent>

            <TabsContent value="jobs" className="space-y-4">
              {isLoadingJobs ? (
                <div className="flex items-center justify-center py-12">
                  <div className="flex flex-col items-center gap-2">
                    <Loader2 className="h-8 w-8 animate-spin text-primary" />
                    <p className="text-sm text-muted-foreground">Loading ingestion jobs...</p>
                  </div>
                </div>
              ) : jobs.length === 0 ? (
                <Card>
                  <CardContent className="flex flex-col items-center justify-center py-12">
                    <PlayCircle className="h-12 w-12 text-muted-foreground mb-4" />
                    <p className="text-muted-foreground">No ingestion jobs found</p>
                    <p className="text-sm text-muted-foreground mt-2">
                      Start an ingestion job from the Websites tab
                    </p>
                  </CardContent>
                </Card>
              ) : (
                <div className="grid gap-4">{jobs.slice(0, 20).map((job) => {
                  const website = websites.find((w) => w.id === job.website_id);
                  const org = organizations.find((o) => o.id === website?.org_id);
                  
                  return (
                    <Card key={job.id}>
                      <CardHeader>
                        <div className="flex items-start justify-between">
                          <div className="flex-1">
                            <CardTitle className="text-base">
                              {website?.url || 'Unknown Website'}
                            </CardTitle>
                            <CardDescription>
                              {org?.name || 'Unknown Organization'}
                            </CardDescription>
                          </div>
                          <Badge variant={job.status === 'completed' ? 'default' : job.status === 'failed' ? 'destructive' : 'secondary'} className="gap-1">
                            {getStatusIcon(job.status)}
                            {job.status}
                          </Badge>
                        </div>
                      </CardHeader>
                      <CardContent className="space-y-3">
                        {job.progress_percent !== undefined && job.status === 'running' && (
                          <div className="space-y-1">
                            <div className="flex justify-between text-sm">
                              <span className="text-muted-foreground">Progress</span>
                              <span className="font-medium">{job.progress_percent}%</span>
                            </div>
                            <Progress value={job.progress_percent} />
                          </div>
                        )}

                        <div className="flex gap-6 text-sm">
                          {job.pages_crawled !== undefined && (
                            <div>
                              <span className="text-muted-foreground">Pages: </span>
                              <span className="font-medium">{job.pages_crawled}</span>
                            </div>
                          )}
                          {job.chunks_created !== undefined && (
                            <div>
                              <span className="text-muted-foreground">Chunks: </span>
                              <span className="font-medium">{job.chunks_created}</span>
                            </div>
                          )}
                        </div>

                        {job.error_message && (
                          <div className="text-sm text-destructive">
                            Error: {job.error_message}
                          </div>
                        )}

                        <div className="text-xs text-muted-foreground">
                          Created: {new Date(job.created_at).toLocaleString()}
                        </div>
                      </CardContent>
                    </Card>
                  );
                })}</div>
              )}
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </ProtectedRoute>
  );
}
