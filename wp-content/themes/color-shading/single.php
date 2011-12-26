<?php get_header(); ?>
			<div id="gray-shading" class="container">
				<div class="padding">

					<div id="content">
                            
                      <?php if(have_posts()) : while(have_posts()) : the_post(); ?>

						<div class="gutter">
                        
                          <div <?php post_class(); ?> id="post-<?php the_ID(); ?>">

							<!-- start content -->

							<h2><?php the_title(); ?></h2>
									   
							<?php the_content(); ?>
							<br class="clear" />
                            <?php wp_link_pages('before=<p id="page-links">'.__('Pages:','cover-wp').' &after=</p>'); ?>
                            
                            <div class="postmetadata alt">
					<small>
						This entry was posted
						<?php /* This is commented, because it requires a little adjusting sometimes.
							You'll need to download this plugin, and follow the instructions:
							http://binarybonsai.com/wordpress/time-since/ */
							/* $entry_datetime = abs(strtotime($post->post_date) - (60*120)); echo time_since($entry_datetime); echo ' ago'; */ ?>
						on <?php the_time('l, F jS, Y') ?> at <?php the_time() ?>,
						is filed under <?php the_category(', ') ?>, and is tagged with <?php the_tags('',', ','') ?>.
						You can follow any responses to this entry through the <?php post_comments_feed_link('RSS 2.0'); ?> feed.

						<?php if ( comments_open() && pings_open() ) {
							// Both Comments and Pings are open ?>
							You can <a href="#respond">leave a response</a>, or <a href="<?php trackback_url(); ?>" rel="trackback">trackback</a> from your own site.

						<?php } elseif ( !comments_open() && pings_open() ) {
							// Only Pings are Open ?>
							Responses are currently closed, but you can <a href="<?php trackback_url(); ?> " rel="trackback">trackback</a> from your own site.

						<?php } elseif ( comments_open() && !pings_open() ) {
							// Comments are open, Pings are not ?>
							You can skip to the end and leave a response. Pinging is currently not allowed.

						<?php } elseif ( !comments_open() && !pings_open() ) {
							// Neither Comments, nor Pings are open ?>
							Both comments and pings are currently closed.

						<?php } edit_post_link('Edit this entry','','.'); ?>

					</small>
				</div>
							
							<!-- end content -->
							
							<br class="clear" />
                            
                          </div>
							
						</div>
                            
						<div class="gutter comments">

                            <?php comments_template(); ?>

                        </div>

                      <?php endwhile; endif; ?>

					</div>
                    
				</div>
			</div>

<?php get_footer(); ?>